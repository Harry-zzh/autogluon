import logging
from typing import Optional

import torch
from torch import nn

from ...constants import AUTOMM, FEATURES, LABEL, LOGITS, WEIGHT
from ..custom_transformer import CLSToken, Custom_Transformer
from ..utils import init_weights, run_model, create_adaptation
from .base import AbstractMultimodalFusionModel
from transformers import LlamaForSequenceClassification, BitsAndBytesConfig
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
import re
from .augment_network import AugmentNetwork
from omegaconf import OmegaConf, DictConfig
import torch.nn.functional as F
logger = logging.getLogger(__name__)

# aug loss
def consist_loss(p_logits, q_logits, threshold):
    p = F.softmax(p_logits, dim=1)
    logp = F.log_softmax(p_logits, dim=1)
    logq = F.log_softmax(q_logits, dim=1)
    loss = torch.sum(p * (logp - logq), dim=-1)
    q = F.softmax(q_logits, dim=1)
    q_largest = torch.max(q, dim=1)[0]
    loss_mask = torch.gt(q_largest, threshold).float()
    loss = loss * loss_mask
    return torch.mean(loss)

# alignment loss
def cal_alignment_loss(p_logits, q_logits):
    if p_logits.size()[-1] == 1: # regression
        mse_loss = nn.MSELoss()
        return mse_loss(p_logits, q_logits)
    else:
        kl_loss = nn.KLDivLoss(reduction="batchmean", log_target = True)
        # input should be a distribution in the log space
        input = F.log_softmax(p_logits, dim=1)
        # Sample a batch of distributions. Usually this would come from the dataset
        target = F.log_softmax(q_logits, dim=1)
        return kl_loss(input, target)



class MultimodalFusionTransformer(AbstractMultimodalFusionModel):
    """
    Use Transformer to fuse different models' features (single-modal and multimodal).
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through Transformer.
    """

    def __init__(
        self,
        prefix: str,
        models: list,
        hidden_features: int,
        num_classes: int,
        n_blocks: Optional[int] = 0,
        attention_n_heads: Optional[int] = 8,
        attention_initialization: Optional[str] = "kaiming",
        attention_normalization: Optional[str] = "layer_norm",
        attention_dropout: Optional[str] = 0.2,
        residual_dropout: Optional[str] = 0.0,
        ffn_activation: Optional[str] = "reglu",
        ffn_normalization: Optional[str] = "layer_norm",
        ffn_d_hidden: Optional[str] = 192,
        ffn_dropout: Optional[str] = 0.0,
        prenormalization: Optional[bool] = True,
        first_prenormalization: Optional[bool] = False,
        kv_compression_ratio: Optional[float] = None,
        kv_compression_sharing: Optional[str] = None,
        head_activation: Optional[str] = "relu",
        head_normalization: Optional[str] = "layer_norm",
        adapt_in_features: Optional[str] = None,
        loss_weight: Optional[float] = None,
        additive_attention: Optional[bool] = False,
        share_qv_weights: Optional[bool] = False,
        use_llama: Optional[bool] = False,
        use_llama_7B: Optional[bool] = False,
        llama_7B_token: Optional[str] = None,
        use_contrastive_loss:  Optional[bool] = False,
        alignment_loss: Optional[str] = None,
        aug_config: Optional[DictConfig] = None,
    ):
        """
        Parameters
        ----------
        prefix
            The fusion model's prefix
        models
            The individual models whose output features will be fused.
        hidden_features
            A list of integers representing the hidden feature dimensions. For example,
            [512, 128, 64] indicates three hidden MLP layers with their corresponding output
            feature dimensions.
        num_classes
            The number of classes.
        n_blocks
            Number of the `FT_Transformer` blocks, which should be non-negative.
        attention_n_heads
            Number of attention heads in each `FT_Transformer` block, which should be positive.
        attention_dropout
            Dropout ratio for the Multi Headed Attention module.
        attention_initialization
            Weights initialization scheme for Multi Headed Attention module.
        attention_normalization
            Normalization policy for attention layers. "layer_norm" is a good default.
        residual_dropout
            Dropout ratio for the linear layers in FT_Transformer block.
        ffn_d_hidden
            Number of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_dropout
            Dropout ratio of the hidden nodes of the linear layers in the Feed-Forward Network module.
        ffn_activation
            Activation function type for the Feed-Forward Network module.
        ffn_normalization
            Normalization scheme of the Feed-Forward Network module.
        prenormalization, first_prenormalization
            Prenormalization to stabilize the training.
        kv_compression_ratio
            The compression ration to reduce the input sequence length.
        kv_compression_sharing
            If `true` the projections will share weights.
        head_activation
            Activation function type of the MLP layer.
        head_normalization
            Normalization scheme of the MLP layer.
        adapt_in_features
            Choice of how to adapt the features of each model. We now support
            - min
                Adapt all features to the minimum dimension. For example, if three models have
                feature dimensions [512, 768, 64], it will linearly map all the features to
                dimension 64.
            - max
                Adapt all features to the maximum dimension. For example, if three models have
                feature dimensions are [512, 768, 64], it will linearly map all the features to
                dimension 768.
        loss_weight
            The weight of individual models. For example, if we fuse the features of ViT, CLIP, and BERT,
            The loss will be computed as "loss = fusion_loss + loss_weight(vit_loss + clip_loss + bert_loss)".
            Basically, it supports adding an auxiliary loss for each individual model.
        additive_attention
            If 'true' the transformer will use additive attention with linear complexity to sequence length.
        share_qv_weights
            if 'true', then value and query transformation parameters are shared in additive attention.
        """
        super().__init__(
            prefix=prefix,
            models=models,
            loss_weight=loss_weight,
        )
        logger.debug("initializing MultimodalFusionTransformer")
        if loss_weight is not None:
            assert loss_weight > 0

        raw_in_features = [per_model.out_features for per_model in models]

        if use_llama_7B:
            base_in_feat = 4096
        elif adapt_in_features == "min":
            base_in_feat = min(raw_in_features)
        elif adapt_in_features == "max":
            base_in_feat = max(raw_in_features)
        
        else:
            raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

        self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])

        in_features = base_in_feat

        assert len(self.adapter) == len(self.model)

        self.use_llama_7B = use_llama_7B
        if use_llama_7B:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type='nf4'
            )
            # quantization_config
            self.fusion_transformer = LlamaForSequenceClassification.from_pretrained(
            "meta-llama/Llama-2-7b-hf", num_labels=num_classes, quantization_config=quantization_config,token=llama_7B_token,)
            self.fusion_transformer = prepare_model_for_kbit_training(self.fusion_transformer)
            lora_config = LoraConfig(
                r=3,
                target_modules=["q_proj", "k_proj", "v_proj"],
                bias="none"
            )
            self.fusion_transformer = get_peft_model(self.fusion_transformer, peft_config=lora_config)
          
            for k, v in self.fusion_transformer.named_parameters():
                if "score" in k:
                    v.requires_grad = True

        else:
            self.fusion_transformer = Custom_Transformer(
                d_token=in_features,
                n_blocks=n_blocks,
                attention_n_heads=attention_n_heads,
                attention_dropout=attention_dropout,
                attention_initialization=attention_initialization,
                attention_normalization=attention_normalization,
                ffn_d_hidden=ffn_d_hidden,
                ffn_dropout=ffn_dropout,
                ffn_activation=ffn_activation,
                ffn_normalization=ffn_normalization,
                residual_dropout=residual_dropout,
                prenormalization=prenormalization,
                first_prenormalization=first_prenormalization,
                last_layer_query_idx=None,
                n_tokens=None,
                kv_compression_ratio=kv_compression_ratio,
                kv_compression_sharing=kv_compression_sharing,
                head_activation=head_activation,
                head_normalization=head_normalization,
                d_out=hidden_features,
                projection=False,
                additive_attention=additive_attention,
                share_qv_weights=share_qv_weights,
            )

            self.head = Custom_Transformer.Head(
                d_in=in_features,
                d_out=num_classes,
                bias=True,
                activation=head_activation,
                normalization=head_normalization,
            )

        self.cls_token = CLSToken(
            d_token=in_features,
            initialization="uniform",
        )

        self.out_features = in_features

        # init weights
        self.adapter.apply(init_weights)
        if not use_llama and not use_llama_7B:
            self.head.apply(init_weights)

        # Initialize Augmentation Network
        self.augmenter = None
        self.aug_config = aug_config
        if aug_config != None and aug_config.turn_on:
            self.adapter_out_dim = base_in_feat
            self.augmenter = self.construct_augnet()

        self.name_to_id = self.get_layer_ids()
        if use_llama or use_llama_7B:
            for k, v in self.name_to_id.items():
                if "fusion_transformer" in k:
                    if "lora" not in k and "score" not in k:
                        self.name_to_id[k] = v + 1
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]
        
        # alignment loss
        self.alignment_loss = alignment_loss

        
    def construct_augnet(self):
        model_feature_dict = [
            (per_model.prefix, per_model.out_features) for per_model in self.model
        ]
        return AugmentNetwork(
            self.aug_config, model_feature_dict, self.adapter_out_dim, len(self.model)
        )

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self,
        batch: dict,
    ):
        multimodal_features = []
        output = {}
        for per_model, per_adapter in zip(self.model, self.adapter):
            per_output = run_model(per_model, batch)
            multimodal_feature = per_adapter(per_output[per_model.prefix][FEATURES])
            if multimodal_feature.ndim == 2:
                multimodal_feature = torch.unsqueeze(multimodal_feature, dim=1)
            multimodal_features.append(multimodal_feature)

            if self.loss_weight is not None:
                per_output[per_model.prefix].update(
                    {WEIGHT: torch.tensor(self.loss_weight).to(multimodal_features[0])}
                )
                output.update(per_output)

        alignment_loss = None
        if self.alignment_loss == "positive-only" or self.alignment_loss == "all":
            alignment_loss = 0.
            num = 0
            for i in range(len(multimodal_features)):
                for j in range(len(multimodal_features)):
                    if i == j: continue
                    alignment_loss += cal_alignment_loss(multimodal_features[i], multimodal_features[j])
                    num += 1
            alignment_loss = alignment_loss / num / num 
            


        multimodal_features = torch.cat(multimodal_features, dim=1)
        multimodal_features = self.cls_token(multimodal_features)
        # pass through augmentation network after adapter
        aug_loss = None

        if self.training:
            if self.augmenter is not None:
                # train augment network
                aug_loss = {}
                detached_feature = multimodal_features.detach().clone()  # [bs, dim]

                new, m, v = self.augmenter(detached_feature)
                regularize_loss = self.augmenter.l2_regularize(detached_feature, new)
                KLD_loss = (
                    self.augmenter.kld(m, v) / new.size()[0] / self.aug_config.z_dim
                )

                with torch.no_grad():
                    ori_logits = self.head(self.fusion_transformer(detached_feature))
                aug_logits = self.head(self.fusion_transformer(new))
                consist_reg = consist_loss(
                    aug_logits, ori_logits.detach(), self.aug_config.consist_t
                )
                aug_loss.update(
                    {
                        "consist_loss": consist_reg,
                        "cons_weight": self.aug_config.consist_loss_weight,
                        "regularizer": regularize_loss,
                        "reg_weight": self.aug_config.regularizer_loss_weight,
                        "KLD_loss": KLD_loss,
                        "kl_weight": self.aug_config.kl_loss_weight,
                    }
                )

                after_augment = new.clone()
                after_augment.register_hook(
                    lambda grad: -grad * (self.aug_config.adv_weight)
                )
                multimodal_features = torch.cat(
                    [multimodal_features, after_augment], dim=0
                )

        if not self.use_llama_7B:
            features = self.fusion_transformer(multimodal_features)
            logits = self.head(features)
        else:
            outputs = self.fusion_transformer(input_ids=None,inputs_embeds=multimodal_features)
            logits = outputs.logits
            features = outputs.hidden_states
            
        fusion_output = {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }

        if alignment_loss != None:
            fusion_output[self.prefix].update({"alignment_loss": alignment_loss}) 
        
        if aug_loss is not None:
            fusion_output[self.prefix].update({"augmenter": aug_loss})

        if self.loss_weight is not None:
            fusion_output[self.prefix].update({WEIGHT: torch.tensor(1.0).to(logits)})
            output.update(fusion_output)
            return output
        else:
            return fusion_output
