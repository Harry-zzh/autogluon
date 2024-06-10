import logging
from typing import Optional

import torch
from torch import nn
import re
from ..constants import (
    AUTOMM,
    COLUMN,
    COLUMN_FEATURES,
    FEATURES,
    IMAGE,
    IMAGE_VALID_NUM,
    LABEL,
    LOGIT_SCALE,
    LOGITS,
    MASKS,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
)
import numpy as np
from .utils import (
    assign_layer_ids,
    get_column_features,
    get_hf_config_and_model,
    get_pretrained_tokenizer,
    init_weights,
)

logger = logging.getLogger(__name__)


class CLIPForImageText_fusionmlp(nn.Module):
    """
    Support the CLIP model. Use fusion MLP afterward.
    Refer to https://huggingface.co/docs/transformers/model_doc/clip
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: Optional[int] = None,
        pretrained: Optional[bool] = True,
        tokenizer_name: Optional[str] = "clip",
        use_miss_token_embed: bool = False,
        num_image_columns: Optional[int] = None,
        num_text_columns: Optional[int] = None,
        manifold_mixup: bool = False,
        manifold_mixup_a: float = 0.,
    ):
        """
        Load the pretrained CLIP from huggingface transformers.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint.
        num_classes
            The number of classes. 1 for a regression task.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        tokenizer_name
            Name of the huggingface tokenizer type.
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes

        self.config, self.model = get_hf_config_and_model(checkpoint_name=checkpoint_name, pretrained=pretrained) # 默认是clip-base
        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_pretrained_tokenizer(
            tokenizer_name=self.tokenizer_name,
            checkpoint_name=self.checkpoint_name,
        )

        self.out_features = self.model.config.projection_dim

        self.head = nn.Linear(self.out_features, num_classes) if num_classes else nn.Identity()
        self.head.apply(init_weights)

        self.prefix = prefix
        self.prefix_dict = [prefix+"_IMAGE", prefix+"_TEXT"]

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

        # if use_miss_token_embed:
        #     # use_miss_token_embed_forward = forward_for_miss_token.__get__(self.model, self.model.__class__)
        #     # setattr(self.model, "forward", use_miss_token_embed_forward)
        #     self.model.miss_token_embed = nn.Embedding(1, 512)
        # else:
        self.model.miss_token_embed = None
        if num_text_columns == 0:
            for k, v in self.model.named_parameters():
                if "text_model" in k:
                    v.requires_grad = False
        if num_image_columns == 0:
            for k, v in self.model.named_parameters():
                if "vision_model" in k or "visual" in k:
                    v.requires_grad = False
        
        # for manifold-mixup
        self.manifold_mixup = manifold_mixup
        self.text_module_list = []
        self.visual_module_list = []
        if manifold_mixup:
            self.manifold_mixup_a = manifold_mixup_a
            self.manifold_mixup_indices = None
            self.manifold_mixup_lam = None
            self.module_list = []
            for n, m in self.model.named_modules():
                #if 'conv' in n:
                pattern_t = r"text_model.encoder.layers\.\d+"
                match = re.search(pattern_t, n)
                if match:
                    result = match.group()
                    if result == n:
                        self.text_module_list.append(m)
                else:
                    pattern_v = r"vision_model.encoder.layers\.\d+"
                    match = re.search(pattern_v, n)
                    if match:
                        result = match.group()
                        if result == n:
                            self.visual_module_list.append(m)
                    else: 
                        continue
        print()
        # self.model.miss_token_embed = use_miss_token_embed

    @property
    def text_token_ids_key(self):
        return f"{self.prefix}_{TEXT_TOKEN_IDS}"

    @property
    def text_valid_length_key(self):
        return f"{self.prefix}_{TEXT_VALID_LENGTH}"

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def text_column_prefix(self):
        return f"{self.text_token_ids_key}_{COLUMN}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def text_feature_dim(self):
        return self.model.config.text_config.hidden_size

    @property
    def image_feature_dim(self):
        return self.model.config.vision_config.hidden_size

    @property
    def input_keys(self):
        return [self.text_token_ids_key, self.text_valid_length_key, self.image_key, self.image_valid_num_key, self.label_key]


    def forward(
        self,
        batch: dict,
    ):
        """
        Parameters
        ----------
        batch
            A dictionary containing the input mini-batch data.
            We need to use the keys with the model prefix to index required data.

        Returns
        -------
            A dictionary with logits and features.
        """
        has_image = (self.image_key in batch and batch[self.image_key] != None)
        has_text = (self.text_token_ids_key in batch and batch[self.text_token_ids_key] != None)
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}

        if has_image:
            images = batch[self.image_key]
            image_valid_num = batch[self.image_valid_num_key]
            assert images.dim() == 5
            b, n, c, h, w = images.shape
            if self.training and self.manifold_mixup:
                alpha = self.manifold_mixup_a
                if self.manifold_mixup_lam == None:
                    if alpha <= 0:
                        self.manifold_mixup_lam = 1
                    else:
                        self.manifold_mixup_lam = np.random.beta(alpha, alpha)
                # k = np.random.randint(-1, len(self.module_list))
                k = np.random.randint(0, len(self.visual_module_list))
                if self.manifold_mixup_indices == None:
                    self.manifold_mixup_indices = torch.randperm(images.size(0)).to(images.device)
                modifier_hook = self.visual_module_list[k].register_forward_hook(self.hook_modify)
                vision_outputs = self.model.vision_model(
                    pixel_values=images.reshape((b * n, c, h, w)),
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                modifier_hook.remove()
            else:
                vision_outputs = self.model.vision_model(
                    pixel_values=images.reshape((b * n, c, h, w)),
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            image_features = self.model.visual_projection(vision_outputs.pooler_output)
            steps = torch.arange(0, n).type_as(image_valid_num)
            image_masks = (steps.reshape((1, -1)) < image_valid_num.reshape((-1, 1))).type_as(image_features)  # (b, n)
            image_features = image_features.reshape((b, n, -1)) * image_masks[:, :, None]  # (b, n, num_features)

            # normalized features
            # image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # 默认除以二范数
            image_features = image_features / torch.clamp(image_features.norm(dim=-1, keepdim=True), min=1e-6)  # (b, num_features)

            # collect image features by image column names
            image_column_features, image_column_feature_masks = get_column_features(
                batch=batch,
                column_name_prefix=self.image_column_prefix,
                features=image_features,
                valid_lengths=image_valid_num,
            )
            ret[COLUMN_FEATURES][FEATURES].update(image_column_features)
            ret[COLUMN_FEATURES][MASKS].update(image_column_feature_masks)

            image_features = image_features.mean(dim=1)  # (b, num_features)
            ret[FEATURES+"_IMAGE"] = image_features

        if has_text:
            text_token_ids = batch[self.text_token_ids_key]
            text_valid_length = batch[self.text_valid_length_key]
            steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
            text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)
            assert torch.equal(text_valid_length, text_masks.sum(dim=-1))

            if self.training and self.manifold_mixup:
                alpha = self.manifold_mixup_a
                if self.manifold_mixup_lam == None:
                    if alpha <= 0:
                        self.manifold_mixup_lam = 1
                    else:
                        self.manifold_mixup_lam = np.random.beta(alpha, alpha)
                # k = np.random.randint(-1, len(self.module_list))
                k = np.random.randint(0, len(self.text_module_list))
                if self.manifold_mixup_indices == None:
                    self.manifold_mixup_indices = torch.randperm(text_token_ids.size(0)).to(text_token_ids.device)
                modifier_hook = self.text_module_list[k].register_forward_hook(self.hook_modify)
                text_outputs = self.model.text_model(
                    input_ids=text_token_ids,
                    attention_mask=text_masks,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
                modifier_hook.remove()
            else:
                text_outputs = self.model.text_model(
                    input_ids=text_token_ids,
                    attention_mask=text_masks,
                    output_attentions=True,
                    output_hidden_states=True,
                    return_dict=True,
                )
            text_features = self.model.text_projection(text_outputs.pooler_output)  # (b, num_features)

            # normalized features
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            # collect text features by text column names
            text_column_features, text_column_feature_masks = get_column_features(
                batch=batch,
                column_name_prefix=self.text_column_prefix,
                features=self.model.text_projection(text_outputs.last_hidden_state),
                valid_lengths=text_valid_length,
                cls_feature=text_features,
            )
            ret[COLUMN_FEATURES][FEATURES].update(text_column_features)
            ret[COLUMN_FEATURES][MASKS].update(text_column_feature_masks)
            ret[FEATURES+"_TEXT"] = text_features

        if has_image and has_text:
            if self.num_classes:
                features = image_features + text_features
                logits = self.head(features)
                ret[FEATURES] = features
            else:
                # cosine similarity as logits
                logits = torch.sum(image_features * text_features, dim=-1)

            ret[LOGITS] = logits

        ret[LOGIT_SCALE] = self.model.logit_scale.exp()

        if self.manifold_mixup and self.training:
            ret["manifold_mixup_lam"] = self.manifold_mixup_lam
            ret["manifold_mixup_indices"] = self.manifold_mixup_indices

        # return {self.prefix: ret}
        res = {self.prefix: ret}
        if has_image:
            res[self.prefix_dict[0]] = {FEATURES: ret[FEATURES+"_IMAGE"]}
        if has_text:
            res[self.prefix_dict[1]] = {FEATURES: ret[FEATURES+"_TEXT"]}
        return res
        # return {self.prefix: ret, self.prefix_dict[0]: {FEATURES: ret[FEATURES+"_IMAGE"]}, self.prefix_dict[1]: {FEATURES: ret[FEATURES+"_TEXT"]}}

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefixes = ["model.text_model", "model.vision_model", "model"]
        # later model prefixes can't starts with the early ones
        for i, model_pre in enumerate(model_prefixes):
            for model_pre2 in model_prefixes[i + 1 :]:
                if model_pre2.startswith(model_pre):
                    raise ValueError(
                        f"{model_pre} is a substring of {model_pre2}. Need to swap them in {model_prefixes}."
                    )

        pre_encoder_patterns = ("embeddings", "pre")
        post_encoder_patterns = ("head", "final", "post", "logit", "project")
        names = [n for n, _ in self.named_parameters()]

        name_to_id = {}
        for per_prefix in model_prefixes:
            per_model_name_to_id, names = assign_layer_ids(
                names=names,
                pre_encoder_patterns=pre_encoder_patterns,
                post_encoder_patterns=post_encoder_patterns,
                model_pre=per_prefix,
            )
            name_to_id.update(per_model_name_to_id)

        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

    def hook_modify(self, module, input, output):
        new_output_0 = self.manifold_mixup_lam * output[0] + (1 - self.manifold_mixup_lam) * output[0][self.manifold_mixup_indices]
        return (new_output_0, output[1])