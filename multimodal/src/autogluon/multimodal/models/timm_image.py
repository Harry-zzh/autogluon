import json
import logging
import os
from typing import Dict, List, Optional

import torch
from timm import create_model
from timm.layers.linear import Linear
from torch import nn

from ..constants import AUTOMM, COLUMN, COLUMN_FEATURES, FEATURES, IMAGE, IMAGE_VALID_NUM, LABEL, LOGITS, MASKS
from .utils import assign_layer_ids, get_column_features, get_model_head
from timm.models._manipulate import checkpoint_seq, named_apply
from timm.models.swin_transformer import window_partition, window_reverse
import math
logger = logging.getLogger(__name__)


# Stores the class names of the timm backbones that support variable input size. You can add more backbones to the list.
SUPPORT_VARIABLE_INPUT_SIZE_TIMM_CLASSES = {"convnext", "efficientnet", "mobilenetv3", "regnet", "resnet"}

def forward_sequential_fusion(self, x, state = None):
    if state != None:
        x, state = self.forward_features(x, state)
        B, L, C = x.shape
        H = W = int(math.sqrt(L))
        x = x.reshape(B, H, W, C)
    else:
        x = self.forward_features(x, state)
    x = self.forward_head(x)
    if state != None:
        return x, state
    else:
        return x
    
def forward_feature_sequential_fusion(self, x, state= None):
    x = self.patch_embed(x)
    B, H, W, C = x.size()
    
    for layer in self.layers[0:-1]:
        x = layer(x)

    # if state != None: # 只弄最后一个stage？
    #     x = x.reshape(B, -1, C)
    #     state = state.expand(B, -1, -1)
    #     x = torch.cat((state, x), dim=1)
    x = self.layers[-1](x, state)

    x = self.norm(x)
    if state != None:
        state = x[:, 0, :]
        x = x[:, 1:, :]
        return x, state
    return x

def forward_transformer_stage_sequential_fusion(self, x, state =None): 
    # if x.dim() == 3: # 有state
    #     state = x[:, 0, :]
    #     x = x[:, 1:, :]
    # else:
    #     state = None
    
    x = self.downsample(x)

    B, H, W, C = x.size()
    if state != None:
        x = x.reshape(B, -1, C)
        state = state.unsqueeze(1)
        x = torch.cat((state, x), dim=1)

    if self.grad_checkpointing and not torch.jit.is_scripting():
        x = checkpoint_seq(self.blocks, x)
    else:
        x = self.blocks(x)
    return x

def forward_block_sequential_fusion(self, x):
    x = self.norm1(x)
    x = x + self.drop_path1(self._attn(x))
    x = x + self.drop_path2(self.mlp(self.norm2(x)))
    return x


def forward_attn_sequential_fusion(self, x):
    if x.dim() == 3: # 用了sequential fusion state
        B, L, C = x.shape
        state = x[:, 0, :]
        x = x[:, 1:, :]
        H = W = int(math.sqrt(L - 1))
        x = x.reshape(B, H, W ,C)
    else:
        B, H, W, C = x.shape
        state = None

    # cyclic shift
    has_shift = any(self.shift_size)
    if has_shift:
        shifted_x = torch.roll(x, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
    else:
        shifted_x = x

    # pad for resolution not divisible by window size
    pad_h = (self.window_size[0] - H % self.window_size[0]) % self.window_size[0]
    pad_w = (self.window_size[1] - W % self.window_size[1]) % self.window_size[1]
    shifted_x = torch.nn.functional.pad(shifted_x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    # partition windows
    x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
    x_windows = x_windows.view(-1, self.window_area, C)  # nW*B, window_size*window_size, C

    # 把state concate回去
    if state != None:
        # B, num_prompts, C --> nW*B, num_prompts, C
        num_windows = x_windows.size()[0] // B
        state = state.unsqueeze(1)
        state = state.expand(num_windows, -1, -1, -1)
        state = state.reshape((-1, 1, C))
        x_windows = torch.cat((state, x_windows), dim=1)

    # W-MSA/SW-MSA
    attn_windows = self.attn(x_windows, mask=self.attn_mask, state=state)  # nW*B, window_size*window_size, C

    if state != None: # 用了sequential fusion state
        state = attn_windows[:, 0, :].view(-1, B, 1, C)
        attn_windows = attn_windows[:, 1:, :]
        state = state.mean(0)

    # merge windows
    attn_windows = attn_windows.view(-1, self.window_size[0], self.window_size[1], C)
    shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # B H' W' C
    shifted_x = shifted_x[:, :H, :W, :].contiguous()

    # reverse cyclic shift
    if has_shift:
        x = torch.roll(shifted_x, shifts=self.shift_size, dims=(1, 2))
    else:
        x = shifted_x

    if state != None:
        x = x.view(B, -1, C)
        x = torch.cat((state, x), dim=1)

    return x

def forward_window_attn_seq_fusion(self, x, mask: Optional[torch.Tensor] = None, state = None):
    B_, N, C = x.shape
    qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    if self.fused_attn:
        attn_mask = self._get_rel_pos_bias()
        if mask is not None:
            num_win = mask.shape[0]
            mask = mask.view(1, num_win, 1, N, N).expand(B_ // num_win, -1, self.num_heads, -1, -1)
            attn_mask = attn_mask + mask.reshape(-1, self.num_heads, N, N)
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop.p if self.training else 0.,
        )
    else:
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self._get_rel_pos_bias().squeeze(0)
        _C, _H, _W = relative_position_bias.shape
        if state != None:
            relative_position_bias = torch.cat((
                torch.zeros(_C, 1, _W, device=attn.device),
                relative_position_bias
                ), dim=1)
            relative_position_bias = torch.cat((
                torch.zeros(_C, _H + 1, 1, device=attn.device),
                relative_position_bias
                ), dim=-1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_win = mask.shape[0]
            if state != None:
                # expand relative_position_bias
                mask = torch.cat((
                    torch.zeros(num_win, 1, _W, device=attn.device),
                    mask), dim=1)
                mask = torch.cat((
                    torch.zeros(
                        num_win, _H + 1, 1,
                        device=attn.device),
                    mask), dim=-1)
            attn = attn.view(-1, num_win, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        x = attn @ v

    x = x.transpose(1, 2).reshape(B_, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


class TimmAutoModelForImagePrediction(nn.Module):
    """
    Support TIMM image backbones.
    Refer to https://github.com/rwightman/pytorch-image-models
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str,
        num_classes: Optional[int] = 0,
        mix_choice: Optional[str] = "all_logits",
        pretrained: Optional[bool] = True,
        early_fusion = False,
        sequential_fusion=False
    ):
        """
        Load a pretrained image backbone from TIMM.

        Parameters
        ----------
        prefix
            The prefix of the TimmAutoModelForImagePrediction model.
        checkpoint_name
            Name of the timm checkpoint, or local parent directory of the saved finetuned timm weights and config.
        num_classes
            The number of classes. 1 for a regression task.
        mix_choice
            Choice used for mixing multiple images. We now support.
            - all_images
                The images are directly averaged and passed to the model.
            - all_logits
                The logits output from individual images are averaged to generate the final output.
        pretrained
            Whether using the pretrained timm models. If pretrained=True, download the pretrained model.
        """
        super().__init__()
        # In TIMM, if num_classes==0, then create_model would automatically set self.model.head = nn.Identity()
        logger.debug(f"initializing {checkpoint_name}")
        if os.path.exists(checkpoint_name):
            checkpoint_path = f"{checkpoint_name}/pytorch_model.bin"
            try:
                with open(f"{checkpoint_name}/config.json") as f:
                    self.config = json.load(f)
                    pretrained_cfg = self.config.get("pretrained_cfg", {})
                    for k, v in pretrained_cfg.items():
                        if k not in self.config:
                            self.config[k] = v
                    self.checkpoint_name = self.config.get("architecture", None)
                    # depths: 2, 2, 18, 2
                    self.model = create_model(self.checkpoint_name, checkpoint_path=checkpoint_path, num_classes=0)
                    # create a head with new num_classes
                    self.head = (
                        Linear(in_features=self.config["num_features"], out_features=num_classes)
                        if num_classes > 0
                        else nn.Identity()
                    )
                    self.num_classes = num_classes if num_classes is not None else 0
            except:
                raise ValueError(f"Timm model path {checkpoint_name} does not exist or model is invalid.")
        else:
            self.checkpoint_name = checkpoint_name
            self.model = create_model(checkpoint_name, pretrained=pretrained, num_classes=num_classes)
            self.head = get_model_head(model=self.model)
            self.config = self.model.default_cfg
            self.num_classes = self.model.num_classes

        self.pretrained = pretrained
        self.out_features = self.model.num_features
        self.global_pool = self.model.global_pool if hasattr(self.model, "global_pool") else None
        self.model.reset_classifier(0)  # remove the internal head

        self.mix_choice = mix_choice
        logger.debug(f"mix_choice: {mix_choice}")

        self.prefix = prefix

        # sequential_fusion 相关
        self.sequential_fusion = sequential_fusion
        if self.sequential_fusion:
            sequential_fusion_forward = forward_sequential_fusion.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward", sequential_fusion_forward)

            fea_sequential_fusion_forward = forward_feature_sequential_fusion.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward_features", fea_sequential_fusion_forward)

            for layer in self.model.layers:
                layer_forward = forward_transformer_stage_sequential_fusion.__get__(layer, layer.__class__)
                setattr(layer, "forward", layer_forward)

                for block in layer.blocks:
                    block_forward = forward_block_sequential_fusion.__get__(block, block.__class__)
                    setattr(block, "forward", block_forward)
                    attn_forward = forward_attn_sequential_fusion.__get__(block, block.__class__)
                    setattr(block, "_attn", attn_forward)

                    window_attn_forward = forward_window_attn_seq_fusion.__get__(block.attn, block.attn.__class__)
                    setattr(block.attn, "forward", window_attn_forward)



        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

        
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
    def input_keys(self):
        return [self.image_key, self.image_valid_num_key]

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    @property
    def image_feature_dim(self):
        return self.model.num_features

    def support_variable_input_size(self):
        """Whether the TIMM image support images sizes that are different from the default used in the backbones"""
        if "test_input_size" in self.config and self.config["test_input_size"] != self.config["input_size"]:
            return True
        cls_name = type(self.model).__name__.lower()
        for k in SUPPORT_VARIABLE_INPUT_SIZE_TIMM_CLASSES:
            if cls_name in k:
                return True
        return False

    def forward(
        self,
        images: torch.FloatTensor,
        image_valid_num: torch.Tensor,
        pre_state=None,
        image_column_names: Optional[List[str]] = None,
        image_column_indices: Optional[List[torch.Tensor]] = None,
        
    ):
        """
        Parameters
        ----------
        images : torch.FloatTensor
            A tensor in [N, C, H, W] layout to represent the images.
        image_valid_num : torch.Tensor
            A tensor that describes valid number of input images.
        image_column_names : list of str, optional
            A list of strings that indicates names of the image columns.
        image_column_indices : list of torch.Tensor, optional
            A list of tensors that indicates start and stop indices of the image columns.

        Returns
        -------
            A dictionary with logits and features.
        """
        if self.mix_choice == "all_images":  # mix inputs
            mixed_images = (
                images.sum(dim=1) / torch.clamp(image_valid_num, min=1e-6)[:, None, None, None]
            )  # mixed shape: (b, 3, h, w)
            features = self.model(mixed_images)
            if self.num_classes > 0:
                logits = self.head(features)
            else:
                logits = features

            column_features = {}
            column_feature_masks = {}

        elif self.mix_choice == "all_logits":  # mix outputs
            b, n, c, h, w = images.shape
            if pre_state != None:
                features, state = self.model(images.reshape((b * n, c, h, w)), state=pre_state)  # (b*n, num_features)
            else:
                features = self.model(images.reshape((b * n, c, h, w))) 
            if self.num_classes > 0:
                logits = self.head(features)
            steps = torch.arange(0, n).type_as(image_valid_num)
            image_masks = (steps.reshape((1, -1)) < image_valid_num.reshape((-1, 1))).type_as(features)  # (b, n)
            features = features.reshape((b, n, -1)) * image_masks[:, :, None]  # (b, n, num_features)

            batch = {
                self.image_key: images,
                self.image_valid_num_key: image_valid_num,
            }
            if image_column_names:
                assert len(image_column_names) == len(image_column_indices), "invalid image column inputs"
                for idx, name in enumerate(image_column_names):
                    batch[name] = image_column_indices[idx]

            # collect features by image column names
            column_features, column_feature_masks = get_column_features(
                batch=batch,
                column_name_prefix=self.image_column_prefix,
                features=features,
                valid_lengths=image_valid_num,
            )

            features = features.sum(dim=1) / torch.clamp(image_valid_num, min=1e-6)[:, None]  # (b, num_features)
            if self.num_classes > 0:
                logits = logits.reshape((b, n, -1)) * image_masks[:, :, None]  # (b, n, num_classes)
                logits = logits.sum(dim=1) / torch.clamp(image_valid_num, min=1e-6)[:, None]  # (b, num_classes)
            else:
                logits = features

        else:
            raise ValueError(f"unknown mix_choice: {self.mix_choice}")

        if column_features == {} or column_feature_masks == {}:
            if pre_state != None:
                return features, logits, state
            return features, logits
        else:
            return features, logits, column_features, column_feature_masks

    def get_output_dict(
        self,
        features: torch.Tensor,
        logits: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        column_features: Optional[Dict[str, torch.Tensor]] = None,
        column_feature_masks: Optional[Dict[str, torch.Tensor]] = None,
    ):
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        if state != None:
            ret["state"] = state

        if column_features != None:
            ret[COLUMN_FEATURES][FEATURES].update(column_features)
            ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)

        ret[FEATURES] = features
        if self.num_classes > 0:
            ret[LOGITS] = logits

        return {self.prefix: ret}

    def get_layer_ids(
        self,
    ):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        Due to different backbone architectures in TIMM, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = ("embed", "cls_token", "stem", "bn1", "conv1")
        post_encoder_patterns = ("head", "norm", "bn2")
        names = [n for n, _ in self.named_parameters()]

        name_to_id, names = assign_layer_ids(
            names=names,
            pre_encoder_patterns=pre_encoder_patterns,
            post_encoder_patterns=post_encoder_patterns,
            model_pre=model_prefix,
        )

        if len(names) > 0:
            logger.debug(f"outer layers are treated as head: {names}")
        for n in names:
            assert n not in name_to_id
            name_to_id[n] = 0

        return name_to_id

    def dump_config(
        self,
        config_path: str,
    ):
        """
        Save TIMM image model configs to a local file.

        Parameters
        ----------
        config_path:
            A file to where the config is written to.
        """
        from ..utils import filter_timm_pretrained_cfg

        config = {}
        pretrained_cfg = filter_timm_pretrained_cfg(self.config, remove_source=True, remove_null=True)
        # set some values at root config level
        config["architecture"] = pretrained_cfg.pop("architecture")
        config["num_classes"] = self.num_classes
        config["num_features"] = self.out_features

        global_pool_type = getattr(self, "global_pool", None)
        if isinstance(global_pool_type, str) and global_pool_type:
            config["global_pool"] = global_pool_type

        config["pretrained_cfg"] = pretrained_cfg

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
            logger.info(f"Timm config saved to {config_path}.")

    def save(self, save_path: str = "./", tokenizers: Optional[dict] = None):
        weights_path = f"{save_path}/pytorch_model.bin"
        torch.save(self.model.state_dict(), weights_path)
        logger.info(f"Model {self.prefix} weights saved to {weights_path}.")
        config_path = f"{save_path}/config.json"
        self.dump_config(config_path)
