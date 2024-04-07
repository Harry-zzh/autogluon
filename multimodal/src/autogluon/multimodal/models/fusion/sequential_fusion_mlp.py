import logging
from typing import List, Optional

import torch
from torch import nn

from ...constants import AUTOMM, FEATURES, LABEL, LOGITS, WEIGHT
from ..mlp import MLP
from ..utils import init_weights, run_model
from .base import AbstractMultimodalFusionModel
from ..clipfusion_mlp import CLIPForImageText_fusionmlp

from omegaconf import OmegaConf, DictConfig
# from .augment_network import AugmentNetwork
import torch.nn.functional as F
from abc import ABC, abstractmethod
from ..timm_image import TimmAutoModelForImagePrediction
logger = logging.getLogger(__name__)

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


class SequentialMultimodalFusionMLP(AbstractMultimodalFusionModel):
    """
    Use MLP to fuse different models' features (single-modal and multimodal).
    Specifically, it adapts the features of each model to specified dimensions,
    concatenates the adapted features, and fuses the features through MLP.
    """

    def __init__(
        self,
        prefix: str,
        models: list,
        hidden_features: List[int],
        num_classes: int,
        adapt_in_features: Optional[str] = None,
        activation: Optional[str] = "gelu",
        dropout_prob: Optional[float] = 0.5,
        normalization: Optional[str] = "layer_norm",
        loss_weight: Optional[float] = None,
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
        activation
            Name of activation function.
        dropout_prob
            Dropout probability.
        normalization
            Name of normalization function.
        loss_weight
            The weight of individual models. For example, if we fuse the features of ViT, CLIP, and BERT,
            The loss will be computed as "loss = fusion_loss + loss_weight(vit_loss + clip_loss + bert_loss)".
            Basically, it supports adding an auxiliary loss for each individual model.
        """
        super().__init__(
            prefix=prefix,
            models=models,
            loss_weight=loss_weight,
        )
        logger.debug("initializing SequentialMultimodalFusionMLP")
        if loss_weight == 0.:
            self.loss_weight = loss_weight = None

        if loss_weight is not None:
            assert loss_weight > 0
        self.num_classes = num_classes

        raw_in_features = [per_model.out_features for per_model in models]
        
        if adapt_in_features is not None: # 默认是max
            if adapt_in_features == "min":
                base_in_feat = min(raw_in_features)
            elif adapt_in_features == "max":
                base_in_feat = max(raw_in_features)
            else:
                raise ValueError(f"unknown adapt_in_features: {adapt_in_features}")

            # 不需要了
            # self.adapter = nn.ModuleList([nn.Linear(in_feat, base_in_feat) for in_feat in raw_in_features])
           

           
            # 虽然clip只有一个model，但是是image 和 text的late fusion
            has_clip = False
            for model_name in models:
                if isinstance(model_name, CLIPForImageText_fusionmlp):
                    has_clip = True
                    break
            if has_clip:
                in_features = base_in_feat * (len(raw_in_features) + 1)
            else:
                in_features = base_in_feat * len(raw_in_features)
        else:
            self.adapter = nn.ModuleList([nn.Identity() for _ in range(len(raw_in_features))])
            in_features = sum(raw_in_features)

        # state token的转换
        # 初始state
        self.init_state = TrainableInitState(base_in_feat)

        self.state_adapter = nn.ModuleList([])
        pre_feat = base_in_feat
        for i in range(len(raw_in_features)): # 最后一个的state token无需转换了。
            self.state_adapter.append(nn.Linear(pre_feat, raw_in_features[i]))
            pre_feat = raw_in_features[i]

        # assert len(self.adapter) == len(self.model)

        # fusion_mlp = []
        # for per_hidden_features in hidden_features:
        #     fusion_mlp.append(
        #         MLP(
        #             in_features=in_features,
        #             hidden_features=per_hidden_features,
        #             out_features=per_hidden_features,
        #             num_layers=1,
        #             activation=activation,
        #             dropout_prob=dropout_prob,
        #             normalization=normalization,
        #         )
        #     )
        #     in_features = per_hidden_features
        # self.fusion_mlp = nn.Sequential(*fusion_mlp)
        # in_features has become the latest hidden size
        # self.head = nn.Linear(in_features, num_classes)
        
        if isinstance(self.model[-1], TimmAutoModelForImagePrediction):
            self.head = nn.Linear(self.model[-1].model.num_features, num_classes)
        else:
            self.head = nn.Linear(raw_in_features[-1], num_classes)

        # Initialize Augmentation Network
        self.augmenter = None
        self.aug_config = aug_config
        if aug_config != None and aug_config.turn_on:
            self.augmenter = self.construct_augnet()

        # init weights
        # self.adapter.apply(init_weights)
        # self.fusion_mlp.apply(init_weights)
        self.head.apply(init_weights)
        
         
        self.out_features = in_features
        self.name_to_id = self.get_layer_ids()
        self.head_layer_names = [n for n, layer_id in self.name_to_id.items() if layer_id == 0]

       

    def construct_augnet(self):
        model_feature_dict = [
            (per_model.prefix, per_model.out_features) for per_model in self.model
        ]
        return AugmentNetwork(
            self.aug_config, model_feature_dict, self.adapter_out_dim, len(self.model)
        )


    @property
    def input_keys(self):
        input_keys = []
        for m in self.model:
            assert hasattr(m, "input_keys"), f"invalid model {type(m)}, which doesn't have a 'input_keys' attribute"
            input_keys += m.input_keys
        return input_keys

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    def forward(
        self,
        *args,
    ):
        """

        Parameters
        ----------
        *args
            A list of torch.Tensor(s) containing the input mini-batch data. The fusion model doesn't need to
            directly access the mini-batch data since it aims to fuse the individual models'
            output features.

        Returns
        -------
        If "loss_weight" is None, it returns dictionary containing the fusion model's logits and
        features. Otherwise, it returns a list of dictionaries collecting all the models' output,
        including the fusion model's.
        """
        multimodal_features = []
        multimodal_logits = []
        offset = 0
    
        pre_state = self.init_state(args[0].size()[0]) # batch size
        cur_model_idx = 0 
        for per_model, per_adapter in zip(self.model, self.state_adapter): # 这里的per_adapter来自self.state_adapter
            per_model_args = args[offset : offset + len(per_model.input_keys)]
            batch = dict(zip(per_model.input_keys, per_model_args))
            # 在batch这里要加一个上一步的state 
            # 它还要算stage change loss
            # 它算metric是拿最后一个的输出
            batch['pre_state'] = per_adapter(pre_state)
            cur_model_idx += 1
            per_output = run_model(per_model, batch)
            pre_state = per_output[per_model.prefix]["state"] # 或许可以各自有个cls token, state只是用于融合之前的模态
            # if hasattr(per_model, "prefix_dict"): # for CLIP model only
            #     for prefix in per_model.prefix_dict:
            #         multimodal_features.append(
            #         per_adapter(per_output[prefix][FEATURES].to(per_adapter.weight.dtype))
            #         )
            # else:
            # features 通常是cls token
            if cur_model_idx == len(self.model):
                multimodal_features = per_output[per_model.prefix][FEATURES] # 取最后一步的state
            
            multimodal_logits.append(per_output[per_model.prefix][LOGITS])
            offset += len(per_model.input_keys)
        
        # multimodal_features = torch.cat(multimodal_features, dim=1)

        # pass through augmentation network after adapter
        aug_loss = None

        if self.training:
            if self.augmenter is not None:
                # train augment network
                aug_loss = {}
                detached_feature = multimodal_features.detach().clone()  # [8, 512]

                new, m, v = self.augmenter(detached_feature)
                regularize_loss = self.augmenter.l2_regularize(detached_feature, new)
                KLD_loss = (
                    self.augmenter.kld(m, v) / new.size()[0] / self.aug_config.z_dim
                )

                with torch.no_grad():
                    ori_logits = self.head(self.fusion_mlp(detached_feature))
                aug_logits = self.head(self.fusion_mlp(new))
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

        # features = self.fusion_mlp(multimodal_features)
        # logits = self.head(features)
        # fusion_output = {
        #     self.prefix: {
        #         LOGITS: logits,
        #         FEATURES: features,
        #     }
        # }
        # if self.loss_weight is not None:
        #     fusion_output[self.prefix].update({WEIGHT: 1})
        #     output.update(fusion_output)
        # else:
        #     output = fusion_output

        # if aug_loss is not None:
        #     output.update({"augmenter": aug_loss})

        # return output
    
        # features = self.fusion_mlp(multimodal_features)
        features = multimodal_features
        logits = self.head(features)

        if aug_loss is not None:
            return features, logits, multimodal_logits, aug_loss
        
        else:
             return features, logits, multimodal_logits

    def get_output_dict(self, features: torch.Tensor, logits: torch.Tensor, multimodal_logits: List[torch.Tensor]):
        fusion_output = {
            self.prefix: {
                LOGITS: logits,
                FEATURES: features,
            }
        }
        if self.loss_weight is not None:
            output = {}
            for per_model, per_logits in zip(self.model, multimodal_logits):
                per_output = {per_model.prefix: {}}
                per_output[per_model.prefix][WEIGHT] = torch.tensor(self.loss_weight).to(per_logits.dtype)
                per_output[per_model.prefix][LOGITS] = per_logits
                output.update(per_output)
            fusion_output[self.prefix].update({WEIGHT: torch.tensor(1.0).to(logits)})
            output.update(fusion_output)
            return output
        else:
            return fusion_output

class TrainableInitState(nn.Module):
    """Trainable initial state"""

    def __init__(self, state_size: int):
        super().__init__()
        self.state_value = nn.Parameter(
            torch.randn((1, state_size))
        )
    
    def forward(self, batch_size):
        init_tensor = torch.tile(self.state_value, [batch_size, 1])

        return init_tensor