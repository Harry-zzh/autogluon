import logging
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import logging as hf_logging
from transformers.models.t5 import T5PreTrainedModel

from ..constants import (
    AUTOMM,
    COLUMN,
    COLUMN_FEATURES,
    FEATURES,
    LABEL,
    LOGITS,
    MASKS,
    TEXT_SEGMENT_IDS,
    TEXT_TOKEN_IDS,
    TEXT_VALID_LENGTH,
)
from .utils import (
    DummyLayer,
    assign_layer_ids,
    get_column_features,
    get_hf_config_and_model,
    get_pretrained_tokenizer,
    init_weights,
)
from transformers.modeling_outputs import  BaseModelOutput

hf_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


def forward_for_sequential_fusion(
    self,
    input_ids: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pre_state: Optional[torch.Tensor] = None,
):
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        input_shape = input_ids.size()
    elif inputs_embeds is not None:
        input_shape = inputs_embeds.size()[:-1]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    device = input_ids.device if input_ids is not None else inputs_embeds.device

    if attention_mask is None:
        if pre_state != None:
            attention_mask = torch.ones(input_shape + 1, device=device)
        else:
            attention_mask = torch.ones(input_shape, device=device)

    if token_type_ids is None:
        token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

    embedding_output = self.embeddings(
        input_ids=input_ids,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        mask=attention_mask,
        inputs_embeds=inputs_embeds,
    )
    if pre_state != None:
        attention_mask = torch.cat((attention_mask, torch.ones((1, 1), device=device)), dim=1)
    if pre_state != None:
        embedding_output = torch.cat((pre_state.unsqueeze(1), embedding_output), dim=1)
    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask,
        output_hidden_states=True,
        output_attentions=output_attentions,
        return_dict=return_dict,
    )
    encoded_layers = encoder_outputs[1]

    if self.z_steps > 1:
        hidden_states = encoded_layers[-2]
        layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
        query_states = encoded_layers[-1]
        rel_embeddings = self.encoder.get_rel_embedding()
        attention_mask = self.encoder.get_attention_mask(attention_mask)
        rel_pos = self.encoder.get_rel_pos(embedding_output)
        for layer in layers[1:]:
            query_states = layer(
                hidden_states,
                attention_mask,
                output_attentions=False,
                query_states=query_states,
                relative_pos=rel_pos,
                rel_embeddings=rel_embeddings,
            )
            encoded_layers.append(query_states)

    sequence_output = encoded_layers[-1]

    if not return_dict:
        return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2) :]

    return BaseModelOutput(
        last_hidden_state=sequence_output,
        hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
        attentions=encoder_outputs.attentions,
    )


class HFAutoModelForTextPrediction(nn.Module):
    """
    Support huggingface text backbones.
    Refer to https://github.com/huggingface/transformers
    """

    def __init__(
        self,
        prefix: str,
        checkpoint_name: str = "microsoft/deberta-v3-base",
        num_classes: Optional[int] = 0,
        pooling_mode: Optional[str] = "cls",
        gradient_checkpointing: Optional[bool] = False,
        low_cpu_mem_usage: Optional[bool] = False,
        pretrained: Optional[bool] = True,
        tokenizer_name: Optional[str] = "hf_auto",
        use_fast: Optional[bool] = True,
        early_fusion: bool = False,
        sequential_fusion: bool = False,
    ):
        """
        Load a pretrained huggingface text transformer backbone.

        Parameters
        ----------
        prefix
            The model prefix.
        checkpoint_name
            Name of the checkpoint or the local directory of a custom checkpoint.
            We support loading checkpoint from
            Huggingface Models list: https://huggingface.co/models
            For example, you may use
                English backbones:
                    - 'microsoft/deberta-v3-base'
                    - 'bert-base-uncased'
                    - 'google/electra-base-discriminator'
                    - 'distilroberta-base'
                Multilingual backbones:
                    - 'microsoft/mdeberta-v3-base'
                    - 'xlm-roberta-base'
        num_classes
            The number of classes. 1 for a regression task.
        pooling_mode
            The pooling mode for the Transformer. Can be "cls", or "mean"
        gradient_checkpointing
            Whether to enable gradient checkpointing
        low_cpu_mem_usage
            Whether to turn on the optimization of reducing the peak CPU memory usage when loading the pretrained model.
        pretrained
            Whether using the pretrained weights. If pretrained=True, download the pretrained model.
        tokenizer_name
            Name of the huggingface tokenizer type.
        use_fast
            Use a fast Rust-based tokenizer if it is supported for a given model.
            If a fast tokenizer is not available for a given model, a normal Python-based tokenizer is returned instead.
            See: https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoTokenizer.from_pretrained.use_fast
        """
        super().__init__()
        logger.debug(f"initializing {checkpoint_name}")
        self.checkpoint_name = checkpoint_name
        self.num_classes = num_classes
        self.early_fusion = early_fusion
        self.sequential_fusion = sequential_fusion
        
        self.config, self.model = get_hf_config_and_model(
            checkpoint_name=checkpoint_name, pretrained=pretrained, low_cpu_mem_usage=low_cpu_mem_usage
        )
        self.tokenizer_name = tokenizer_name
        self.tokenizer = get_pretrained_tokenizer(
            tokenizer_name=self.tokenizer_name,
            checkpoint_name=self.checkpoint_name,
            use_fast=use_fast,
        )

        if self.sequential_fusion:
            sequential_forward = forward_for_sequential_fusion.__get__(self.model, self.model.__class__)
            setattr(self.model, "forward", sequential_forward)


        if isinstance(self.model, T5PreTrainedModel):
            self.is_t5 = True
            # Remove the decoder in T5. We will only use the T5 encoder for extracting the embeddings
            del self.model.decoder
        else:
            self.is_t5 = False

        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            if self.is_t5:
                self.dummy_layer = DummyLayer()

        self.out_features = self.model.config.hidden_size

        if not early_fusion:
            self.head = nn.Linear(self.out_features, num_classes) if num_classes else nn.Identity()
            self.head.apply(init_weights)

        self.prefix = prefix
        self.pooling_mode = pooling_mode

        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

        if hasattr(self.model.config, "type_vocab_size") and self.model.config.type_vocab_size <= 1:
            # Disable segment ids for models like RoBERTa
            self.disable_seg_ids = True
        else:
            self.disable_seg_ids = False

        if early_fusion:
            for n, p in self.model.named_parameters():
                if "embed" not in n:
                    p.requires_grad = False

    @property
    def text_token_ids_key(self):
        return f"{self.prefix}_{TEXT_TOKEN_IDS}"

    @property
    def text_segment_ids_key(self):
        return f"{self.prefix}_{TEXT_SEGMENT_IDS}"

    @property
    def text_valid_length_key(self):
        return f"{self.prefix}_{TEXT_VALID_LENGTH}"

    @property
    def input_keys(self):
        return [self.text_token_ids_key, self.text_segment_ids_key, self.text_valid_length_key]

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def text_column_prefix(self):
        return f"{self.text_token_ids_key}_{COLUMN}"

    @property
    def text_feature_dim(self):
        return self.model.config.hidden_size

    def forward(
        self,
        text_token_ids: torch.Tensor,
        text_segment_ids: torch.Tensor,
        text_valid_length: torch.Tensor,
        pre_state: Optional[torch.Tensor] = None,
        text_column_names: Optional[List[str]] = None,
        text_column_indices: Optional[List[torch.Tensor]] = None,
    ):
        """
        Parameters
        ----------
        text_token_ids : torch.Tensor
            Indices of input sequence tokens in the vocabulary.
        text_segment_ids : torch.Tensor
            Indices of input sequence segments.
        text_valid_length : torch.Tensor
            Valid length of the input text sequence.
        text_column_names : list of str, optional
            Names of the text columns.
        text_column_indices : list of torch.Tensor, optional
            Start and stop indices of the text columns.

        Returns
        -------
            A tuple that contains (pooled_features, logits, column_features, column_feature_masks)
        """
        if self.disable_seg_ids:
            text_segment_ids = None

        steps = torch.arange(0, text_token_ids.shape[1]).type_as(text_valid_length)
        text_masks = (steps.reshape((1, -1)) < text_valid_length.reshape((-1, 1))).type_as(text_token_ids)

        if not self.early_fusion:
            if self.is_t5:
                # For the T5 model, we will only use the encoder to encode the sentence. This is adopted in
                # "Sentence-T5 (ST5): Scalable Sentence Encoders from Pre-trained Text-to-Text Models"
                # (https://aclanthology.org/2022.findings-acl.146.pdf).
                inputs_embeds = self.model.encoder.embed_tokens(text_token_ids)
                if self.gradient_checkpointing:
                    # FIXME(?) This is a hack! We added a DummyLayer to ensure that the
                    #  gradient checkpointing will assign output layer as require_grad=True
                    #  Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
                    inputs_embeds = self.dummy_layer(inputs_embeds)
                outputs = self.model.encoder(
                    inputs_embeds=inputs_embeds,
                    attention_mask=text_masks,
                )
            else:
                if "token_type_ids" in self.tokenizer.model_input_names:
                    outputs = self.model(
                        input_ids=text_token_ids,
                        token_type_ids=text_segment_ids,
                        attention_mask=text_masks,
                        pre_state=pre_state
                    )
                else:
                    outputs = self.model(
                        input_ids=text_token_ids,
                        attention_mask=text_masks,
                    )
            if pre_state != None:
                state = outputs.last_hidden_state[:, 0, :]
                outputs.last_hidden_state = outputs.last_hidden_state[:, 1:, :]
            else:
                state = None
            if self.pooling_mode == "cls" or self.pooling_mode == "all":
                pooled_features = outputs.last_hidden_state[:, 0, :]
            elif self.pooling_mode == "mean":
                pooled_features = (outputs.last_hidden_state * text_masks.unsqueeze(-1)).sum(1)
                sum_mask = text_masks.unsqueeze(-1).sum(1)
                sum_mask = torch.clamp(sum_mask, min=1e-9)
                pooled_features = pooled_features / sum_mask
            else:
                raise NotImplementedError(f"Pooling mode={self.pooling_mode} is not supported.")

            logits = self.head(pooled_features)
            last_hidden_state = outputs.last_hidden_state

            batch = {
                self.text_token_ids_key: text_token_ids,
                self.text_segment_ids_key: text_segment_ids,
                self.text_valid_length_key: text_valid_length,
            }
            if text_column_names:
                assert len(text_column_names) == len(text_column_indices), "invalid text column inputs"
                for idx, name in enumerate(text_column_names):
                    batch[name] = text_column_indices[idx]
            column_features, column_feature_masks = get_column_features(
                batch=batch,
                column_name_prefix=self.text_column_prefix,
                features=last_hidden_state,
                valid_lengths=text_valid_length,
                cls_feature=pooled_features,
            )

            if self.pooling_mode == "all":
                pooled_features = last_hidden_state

            if column_features == {} or column_feature_masks == {}:
                return pooled_features, logits, state
            else:
                return pooled_features, logits, column_features, column_feature_masks, state
        else:
            if self.is_t5:
                # For the T5 model, we will only use the encoder to encode the sentence. This is adopted in
                # "Sentence-T5 (ST5): Scalable Sentence Encoders from Pre-trained Text-to-Text Models"
                # (https://aclanthology.org/2022.findings-acl.146.pdf).
                inputs_embeds = self.model.encoder.embed_tokens(text_token_ids)
                if self.gradient_checkpointing:
                    # FIXME(?) This is a hack! We added a DummyLayer to ensure that the
                    #  gradient checkpointing will assign output layer as require_grad=True
                    #  Reference: https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/9
                    inputs_embeds = self.dummy_layer(inputs_embeds)
                return inputs_embeds, torch.tensor([])
            else:
                input_ids = text_token_ids
                inputs_embeds = None
                attention_mask=text_masks
                position_ids=None
                if "token_type_ids" in self.tokenizer.model_input_names:
                    token_type_ids=text_segment_ids
                else:
                    token_type_ids=None
                    
                if input_ids is not None and inputs_embeds is not None:
                    raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
                elif input_ids is not None:
                    input_shape = input_ids.size()
                elif inputs_embeds is not None:
                    input_shape = inputs_embeds.size()[:-1]
                else:
                    raise ValueError("You have to specify either input_ids or inputs_embeds")

                device = input_ids.device if input_ids is not None else inputs_embeds.device

                if attention_mask is None:
                    attention_mask = torch.ones(input_shape, device=device)
                if token_type_ids is None:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

                embedding_output = self.model.embeddings(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    position_ids=position_ids,
                    mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                )
                return embedding_output, torch.tensor([])
                    
    def get_output_dict(
        self,
        pooled_features: torch.Tensor,
        logits: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        column_features: Optional[Dict[str, torch.Tensor]] = None,
        column_feature_masks: Optional[Dict[str, torch.Tensor]] = None,
        
    ):
        if self.early_fusion:
            return {
                self.prefix: {
                    FEATURES: pooled_features,
                }
            }
        ret = {COLUMN_FEATURES: {FEATURES: {}, MASKS: {}}}
        if column_features != None:
            ret[COLUMN_FEATURES][FEATURES].update(column_features)
            ret[COLUMN_FEATURES][MASKS].update(column_feature_masks)
        ret[LOGITS] = logits
        ret[FEATURES] = pooled_features
        ret["state"] = state
        return {self.prefix: ret}

    def get_layer_ids(self):
        """
        Assign an id to each layer. Layer ids will be used in layer-wise lr decay.
        Basically, id gradually increases when going from the output end to
        the input end. The layers defined in this class, e.g., head, have id 0.

        In the AutoModel scenario, this function may not always return the correct result.
        Thus, you can use "print(json.dumps(name_to_id, indent=2))" to manually check whether
        the layer ids are reasonable.

        Returns
        -------
        A dictionary mapping the layer names (keys) to their ids (values).
        """
        model_prefix = "model"
        pre_encoder_patterns = (
            "embeddings",
            "LayerNorm",
            "wte",
            "wpe",
            "shared.weight",
            "encoder.conv.conv",
            "relative_attention_bias",
            "dummy_layer",
        )
        post_encoder_patterns = ("head", "pooler", "ln_f", "final_layer_norm")
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

    def save(self, save_path: str = "./"):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"Model weights and tokenizer for {self.prefix} are saved to {save_path}.")
