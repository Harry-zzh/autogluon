import logging
from typing import Callable, Dict, List, Optional, Union

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
import torchmetrics
from lightning.pytorch.strategies import DeepSpeedStrategy
from lightning.pytorch.utilities import grad_norm
from torch import nn
from torch.nn.modules.loss import _Loss
from torchmetrics.aggregation import BaseAggregator

from ..constants import LM_TARGET, LOGITS, T_FEW, TEMPLATE_LOGITS, WEIGHT, FEATURES
from ..data.mixup import MixupModule, multimodel_mixup
from ..models.utils import run_model
from .semantic_seg_metrics import COD, Balanced_Error_Rate
from .utils import apply_layerwise_lr_decay, apply_single_lr, apply_two_stages_lr, get_lr_scheduler, get_optimizer, get_augment_network_parameters
import copy 
from .losses import SoftTargetCrossEntropy
logger = logging.getLogger(__name__)

def to_one_hot(inp, num_classes):
    y_onehot = torch.FloatTensor(inp.size(0), num_classes).to(inp.device)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.unsqueeze(1).data, 1)
    return y_onehot

def get_ground_truth(device, num_logits) -> torch.Tensor:
    labels = torch.arange(num_logits, device=device, dtype=torch.long)
    return labels

def get_logits(features_1, features_2, logit_scale=1.):
    logits_per_feature_1 = logit_scale * features_1 @ features_2.T
    logits_per_feature_2 = logit_scale * features_2 @ features_1.T

    return logits_per_feature_1, logits_per_feature_2
    
def contrastive_loss(features, logit_scale=1.):
    total_loss = 0.
    num = 0
    # idx = 0
    key_list = list(features.keys())
    for idx_1 in range(0, len(features)):
        for idx_2 in range(idx_1 + 1, len(features)):
            key_1 = key_list[idx_1]
            key_2 = key_list[idx_2]
            logits_1, logits_2 = get_logits(features[key_1], features[key_2], logit_scale)
            labels = get_ground_truth(logits_1.device, logits_1.shape[0])

            total_loss += (F.cross_entropy(logits_1, labels) + F.cross_entropy(logits_2, labels))
            num += 2
    total_loss /= num

    return total_loss

class LitModule(pl.LightningModule):
    """
    Control the loops for training, evaluation, and prediction. This module is independent of
    the model definition. This class inherits from the Pytorch Lightning's LightningModule:
    https://lightning.ai/docs/pytorch/stable/common/lightning_module.html
    """

    def __init__(
        self,
        model: nn.Module,
        optim_type: Optional[str] = None,
        lr_choice: Optional[str] = None,
        lr_schedule: Optional[str] = None,
        lr: Optional[float] = None,
        lr_decay: Optional[float] = None,
        end_lr: Optional[Union[float, int]] = None,
        lr_mult: Optional[Union[float, int]] = None,
        weight_decay: Optional[float] = None,
        warmup_steps: Optional[int] = None,
        loss_func: Optional[_Loss] = None,
        validation_metric: Optional[torchmetrics.Metric] = None,
        validation_metric_name: Optional[str] = None,
        custom_metric_func: Callable = None,
        test_metric: Optional[torchmetrics.Metric] = None,
        efficient_finetune: Optional[str] = None,
        trainable_param_names: Optional[List] = None,
        mixup_fn: Optional[MixupModule] = None,
        mixup_off_epoch: Optional[int] = 0,
        model_postprocess_fn: Callable = None,
        skip_final_val: Optional[bool] = False,
        track_grad_norm: Optional[Union[int, str]] = -1,
        aug_optimizer: Optional[bool] = False,
        aug_turn_on: Optional[bool] = False,
        aug_lr: Optional[float] = None,
        aug_optim_type: Optional[str] = None,
        aug_weight_decay: Optional[float] = None,
        grad_steps: Optional[int] = 1,
        contra_loss: Optional[str] = "",
        contra_loss_w: Optional[float] = None
    ):
        """
        Parameters
        ----------
        model
            A Pytorch model
        optim_type
            Optimizer type. We now support:
            - adamw
            - adam
            - sgd
        lr_choice
            How to set each layer's learning rate. If not specified, the default is a single
            learnng rate for all layers. Otherwise, we now support two choices:
            - two_stages
                The layers in the pretrained models have a small learning rate (lr * lr_mult),
                while the newly added head layers use the provided learning rate.
            - layerwise_decay
                The layers have decreasing learning rate from the output end to the input end.
                The intuition is that later layers are more task-related, hence larger learning rates.
        lr_schedule
            Learning rate schedule. We now support:
            - cosine_decay
                Linear warmup followed by cosine decay
            - polynomial_decay
                Linear warmup followed by polynomial decay
        lr
            Learning rate.
        lr_decay
            The learning rate decay factor (0, 1). It is used only when lr_choice is "layerwise_decay".
        end_lr
            The final learning rate after decay.
        lr_mult
            The learning rate multiplier (0, 1). It is used only when lr_choice is "two_stages".
        weight_decay
            The weight decay to regularize layer weights' l2 norm.
        warmup_steps
            How many steps to warmup learning rate. If a float (0, 1), it would represent the
            percentage of steps over all the training steps. The actual number is calculated as
            "int(warmup_steps * max_steps)". If an integer, it would be the exact step number.
        loss_func
            A Pytorch loss module, e.g., nn.CrossEntropyLoss().
        validation_metric
            A torchmetrics module used in the validation stage, e.g., torchmetrics.Accuracy().
        validation_metric_name
            Name of validation metric in case that validation_metric is a aggregation metric,
            e.g., torchmetrics.MeanMetric, whose name can't reflect the real metric name.
        custom_metric_func
            A customized metric function in case that torchmetrics doesn't have the metric.
            It is generally used together with torchmetrics' aggregators, e.g., torchmetrics.MeanMetric.
            Refer to https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/aggregation.py
        test_metric
            A torchmetrics module used in the test stage, e.g., torchmetrics.Accuracy().
        efficient_finetune
            Whether to use efficient finetuning strategies. This will be helpful for fast finetuning of large backbones.
            We support options such as:

            - bit_fit (only finetune the bias terms)
            - norm_fit (only finetune the weights in norm layers / bias layer)
            - lora, lora_bias, lora_norm (only finetunes decomposition matrices inserted into model, in combination with either bit_fit or norm_fit)
            - ia3, ia3_bias, ia3_norm (adds vector that scales activations by learned vectors, in combination with either bit_fit or norm_fit)
            - None (do not use efficient finetuning strategies)
        track_grad_norm
            Track the p-norm of gradients during training. May be set to ‘inf’ infinity-norm.
            If using Automatic Mixed Precision (AMP), the gradients will be unscaled before logging them.

        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "validation_metric",
                "test_metric",
                "loss_func",
                "model_postprocess_fn",
                "mixup_fn",
                "trainable_param_names",
            ]
        )
        self.model = model
        self.validation_metric = validation_metric
        self.validation_metric_name = f"val_{validation_metric_name}"
        self.loss_func = loss_func
        self.mixup_fn = mixup_fn
        if isinstance(validation_metric, BaseAggregator) and custom_metric_func is None:
            raise ValueError(
                f"validation_metric {validation_metric} is an aggregation metric,"
                "which must be used with a customized metric function."
            )
        self.custom_metric_func = custom_metric_func
        self.model_postprocess_fn = model_postprocess_fn
        self.trainable_param_names = trainable_param_names if trainable_param_names else []
        self.skip_final_val = skip_final_val
        self.track_grad_norm = track_grad_norm
        if aug_optimizer:
            self.automatic_optimization = False
        # self.hparams.grad_steps = 1
        # self.automatic_optimization = False

        self.contra_loss = contra_loss
        if  self.contra_loss != "" and self.hparams.grad_steps > 1:
            self.automatic_optimization = False
            self.accum_batch, self.accum_features = [], {}
            self.contrastive_loss_w = contra_loss_w
            # self.loss_list = [] 
            self.temp_model = copy.deepcopy(self.model)
            for v in self.temp_model.parameters():
                v.requires_grad = False


        print("self.automatic_optimization: ", self.automatic_optimization)
        

    def _compute_template_loss(
        self,
        per_output: Dict,
        label: torch.Tensor,
    ):
        logits = per_output[TEMPLATE_LOGITS]
        choices_scores = per_output[LOGITS]
        lm_target = per_output[LM_TARGET]

        bs = lm_target.size(0)
        num_choices = lm_target.size(1)

        lm_loss = F.cross_entropy(
            logits[range(bs), label].flatten(0, 1),
            lm_target[range(bs), label].flatten(0, 1),
        )
        if self.model.mc_loss > 0:
            mc_loss = F.cross_entropy(choices_scores, label)
        else:
            mc_loss = 0.0

        if self.model.unlikely_loss > 0:
            cand_loglikely = -F.cross_entropy(logits.flatten(0, 2), lm_target.flatten(0, 2), reduction="none").view(
                bs, num_choices, -1
            )
            cand_loglikely += (lm_target < 0) * -100
            cand_loglikely[range(bs), label] = -100
            unlikely_loss = -torch.log(1 - torch.exp(cand_loglikely) + 1e-2).sum() / (cand_loglikely != -100).sum()
        else:
            unlikely_loss = 0.0

        return lm_loss + mc_loss * self.model.mc_loss + unlikely_loss * self.model.unlikely_loss

    def _compute_loss(
        self,
        output: Dict,
        label: torch.Tensor,
    ):
        loss = 0
        for _, per_output in output.items():
            if _ == "augmenter" or _ == "alignment_loss": continue
            weight = per_output[WEIGHT] if WEIGHT in per_output else 1

            if (
                    _.startswith("fusion")
                    and self.model.training and hasattr(self.model, "aug_config") 
                    and self.model.aug_config.turn_on
                ):

                    loss += (
                        self.loss_func(
                            input=per_output[LOGITS].squeeze(dim=1),
                            target=label.tile((2,)),
                        )
                    )
                    self.log("loss/target", loss, prog_bar=True)
            elif (
                TEMPLATE_LOGITS in per_output and self.model.prefix == T_FEW
            ):  # Do only add template loss if T-Few. #TODO Add compatibility to Fusion models.
                loss += self._compute_template_loss(per_output, label) * weight
            else:
                num_classes = per_output[LOGITS].size()[-1]
                if "manifold_mixup_lam" in per_output.keys():
                    indices = per_output["manifold_mixup_indices"]
                    lam = per_output["manifold_mixup_lam"]
                    if num_classes > 1:
                        label_onehot = to_one_hot(label, num_classes)
                        label_shuffled_onehot = label_onehot[indices]
                        label = label_onehot * lam + label_shuffled_onehot * (1 - lam)
                    else:
                        label = label * lam + label[indices] * (1 - lam)
                elif isinstance(self.loss_func, SoftTargetCrossEntropy):
                    label = to_one_hot(label, num_classes)
                    
                loss += (
                    self.loss_func(
                        input=per_output[LOGITS].squeeze(dim=1),
                        target=label,
                    )
                    * weight
                )
        if "augmenter" in output.keys():
            reg_loss = 0
            kl_loss = 0
            c_loss = 0
            l = output["augmenter"]
            if "KLD_loss" in l.keys():
                kl_loss = l["KLD_loss"] * l["kl_weight"]
            if "regularizer" in l.keys():
                reg_loss = l["regularizer"] * l["reg_weight"]
            if "consist_loss" in l.keys():
                c_loss = l["consist_loss"] * l["cons_weight"]
                self.log("loss/consist", c_loss, prog_bar=True)

            self.log("loss/reg_loss", reg_loss, prog_bar=True)
            self.log("loss/kl_loss", kl_loss, prog_bar=True)

            loss = loss + reg_loss + kl_loss + c_loss
        if "alignment_loss" in output.keys():
            loss = loss + output["alignment_loss"]
            self.log("loss/alignment_loss", output["alignment_loss"], prog_bar=True)
        return loss

    def _compute_metric_score(
        self,
        metric: torchmetrics.Metric,
        custom_metric_func: Callable,
        logits: torch.Tensor,
        label: torch.Tensor,
    ):
        if isinstance(
            metric,
            (
                torchmetrics.classification.BinaryAUROC,
                torchmetrics.classification.BinaryAveragePrecision,
                torchmetrics.classification.BinaryF1Score,
            ),
        ):
            prob = F.softmax(logits.float(), dim=1)
            metric.update(preds=prob[:, 1], target=label)  # only for binary classification
        elif isinstance(metric, BaseAggregator):
            metric.update(custom_metric_func(logits, label))
        else:
            metric.update(logits.squeeze(dim=1).float(), label)

    def _shared_step(
        self,
        batch: Dict,
        use_temp: bool=False # when use contrastive loss and use temp_model
    ):
        if self.contra_loss != "" and use_temp:
            label = batch[self.temp_model.label_key]
            if self.mixup_fn is not None:
                self.mixup_fn.mixup_enabled = self.training & (self.current_epoch < self.hparams.mixup_off_epoch)
                batch, label = multimodel_mixup(batch=batch, model=self.temp_model, mixup_fn=self.mixup_fn)
            output = run_model(self.temp_model, batch)
            loss = self._compute_loss(output=output, label=label)
            return output, loss
        label = batch[self.model.label_key]
        if self.mixup_fn is not None:
            self.mixup_fn.mixup_enabled = self.training & (self.current_epoch < self.hparams.mixup_off_epoch)
            batch, label = multimodel_mixup(batch=batch, model=self.model, mixup_fn=self.mixup_fn)
        output = run_model(self.model, batch)
        loss = self._compute_loss(output=output, label=label)
        return output, loss

    def training_step(self, batch, batch_idx):
        """
        Per training step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the training loss.
        batch_idx
            Index of mini-batch.

        Returns
        -------
        Average loss of the mini-batch data.
        """
        if self.hparams.aug_optimizer:
            if self.hparams.aug_turn_on:
                target_optimizer, aug_optimizer = self.optimizers()
            else:
                target_optimizer = self.optimizers()
            target_opt_scheduler = self.lr_schedulers()

            output, loss = self._shared_step(batch)
            loss = loss / self.hparams.grad_steps
            self.manual_backward(loss)
            
            # gradient accumulation
            if (
                batch_idx + 1
            ) % self.hparams.grad_steps == 0 or self.trainer.is_last_batch:
                # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                target_optimizer.step()
                target_opt_scheduler.step()
                if  "fusion" in self.model.__class__.__name__.lower() and self.hparams.aug_turn_on:
                    aug_optimizer.step()

                target_optimizer.zero_grad()
                if  "fusion" in self.model.__class__.__name__.lower() and self.hparams.aug_turn_on:
                    aug_optimizer.zero_grad()

                    
        elif self.contra_loss != "" and self.hparams.grad_steps > 1:
            optimizer = self.optimizers()
            scheduler = self.lr_schedulers()

            # First, cache the features without any gradient tracking.
            with torch.no_grad():
                output, loss = self._shared_step(batch, use_temp=True)
                # self.loss_list.append(loss)

            for key, per_output in output.items():
                if key.startswith("fusion"): continue
                if key in self.accum_features:
                    if self.contra_loss == "contra_logit":
                        self.accum_features[key].append(per_output[LOGITS])
                    elif self.contra_loss == "contra_fea":
                        self.accum_features[key].append(per_output[FEATURES])
                else:
                    if self.contra_loss == "contra_logit":
                        self.accum_features[key] = [per_output[LOGITS]]
                    elif self.contra_loss == "contra_fea":
                        self.accum_features[key] = [per_output[FEATURES]]

            self.accum_batch.append(batch)
            if not ((batch_idx + 1) % self.hparams.grad_steps == 0):
                # FIXME this makes data time logging unreliable when accumulating
                return
             
            # Now, ready to take gradients for the last accum_freq batches.
            # Re-do the forward pass for those batches, and use the cached features from the other batches as negatives.
            # Call backwards each time, but only step optimizer at the end.
            # self.model.train()
            for j in range(self.hparams.grad_steps):
                output, loss = self._shared_step(self.accum_batch[j])

                inputs = {}
                for key, val in self.accum_features.items():
                    accumulated = self.accum_features[key]
                    if self.contra_loss == "contra_logit":
                        inputs[key] = torch.cat(accumulated[:j] + [output[key][LOGITS]] + accumulated[j + 1:])
                    elif self.contra_loss == "contra_fea":
                        inputs[key] = torch.cat(accumulated[:j] + [output[key][FEATURES]] + accumulated[j + 1:])

                contra_loss = self.contrastive_loss_w * contrastive_loss(inputs)
                self.log("contrastive_loss", contra_loss, prog_bar=True)
                self.log("target_loss", loss, prog_bar=True)
                losses = loss + contra_loss
                total_loss = losses  / self.hparams.grad_steps
                self.manual_backward(total_loss)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            self.temp_model = copy.deepcopy(self.model)
            for v in self.temp_model.parameters():
                v.requires_grad = False

            self.accum_features = {}
            self.accum_batch = []
    
    
        else:
            output, loss = self._shared_step(batch)
            if not self.automatic_optimization:
                target_optimizer  = self.optimizers()
                target_opt_scheduler = self.lr_schedulers()
                loss = loss / self.hparams.grad_steps 
                self.manual_backward(loss)
                # print([(n, p.grad is not None) for n, p in self.named_parameters()])
                # print([n for n, p in self.named_parameters() if p.grad ==None])

                if (batch_idx + 1) % self.hparams.grad_steps == 0 or self.trainer.is_last_batch:
                    # nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                    target_optimizer.step()
                    target_opt_scheduler.step()
                  
                    target_optimizer.zero_grad()

            self.log("train_loss", loss, prog_bar=True)
        return loss

    def on_validation_start(self) -> None:
        if self.skip_final_val and self.trainer.should_stop:
            self.log(
                self.validation_metric_name,
                self.validation_metric,
                on_step=False,
                on_epoch=True,
            )
            return None
        else:
            return super().on_validation_start()

    def validation_step(self, batch, batch_idx):
        """
        Per validation step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data, including both input data and
            ground-truth labels. The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix. The
            ground-truth labels are used here to compute the validation loss and metric.
            The validation metric is used for top k model selection and early stopping.
        batch_idx
            Index of mini-batch.
        """
        output, loss = self._shared_step(batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)
        # By default, on_step=False and on_epoch=True
        self.log("val_loss", loss, prog_bar=True)
        self._compute_metric_score(
            metric=self.validation_metric,
            custom_metric_func=self.custom_metric_func,
            logits=output[self.model.prefix][LOGITS],
            label=batch[self.model.label_key],
        ),
        self.log(
            self.validation_metric_name,
            self.validation_metric,
            on_step=False,
            on_epoch=True,
        )

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Per prediction step. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#prediction-loop

        Parameters
        ----------
        batch
            A dictionary containing the mini-batch data.
            The mini-batch data are passed to each individual model,
            which indexes its required input data by keys with its model prefix.
            Ground-truth labels are not needed for prediction.
        batch_idx
            Index of mini-batch.
        dataloader_idx
            Index of dataloader.
        Returns
        -------
        A dictionary with the mini-batch's logits and features.
        """
        output = run_model(self.model, batch)
        if self.model_postprocess_fn:
            output = self.model_postprocess_fn(output)

        return output[self.model.prefix]

    def configure_optimizers(self):
        """
        Configure optimizer. This function is registered by LightningModule.
        Refer to https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        Returns
        -------
        [optimizer]
            Optimizer.
        [sched]
            Learning rate scheduler.
        """
        kwargs = dict(
            model=self.model,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        if self.hparams.lr_choice == "two_stages":
            logger.debug("applying 2-stage learning rate...")
            grouped_parameters = apply_two_stages_lr(
                lr_mult=self.hparams.lr_mult,
                return_params=True,
                **kwargs,
            )
        elif self.hparams.lr_choice == "layerwise_decay":
            logger.debug("applying layerwise learning rate decay...")
            grouped_parameters = apply_layerwise_lr_decay(
                lr_decay=self.hparams.lr_decay,
                efficient_finetune=self.hparams.efficient_finetune,
                trainable_param_names=self.trainable_param_names,
                seperate_augment_optimizer=self.hparams.aug_optimizer,
                **kwargs,
            )
        else:
            logger.debug("applying single learning rate...")
            grouped_parameters = apply_single_lr(
                efficient_finetune=self.hparams.efficient_finetune,
                trainable_param_names=self.trainable_param_names,
                **kwargs,
            )

        optimizer = get_optimizer(
            optim_type=self.hparams.optim_type,
            optimizer_grouped_parameters=grouped_parameters,
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        logger.debug(f"trainer.max_steps: {self.trainer.max_steps}")
        if self.trainer.max_steps is None or -1:
            if isinstance(self.trainer.strategy, DeepSpeedStrategy):
                max_steps = 1
            else:
                if self.automatic_optimization:
                    max_steps = (
                        len(self.trainer.datamodule.train_dataloader())
                        * self.trainer.max_epochs
                        // self.trainer.accumulate_grad_batches
                    )
                else:
                    max_steps = (
                        len(self.trainer.datamodule.train_dataloader())
                        * self.trainer.max_epochs
                        // self.hparams.grad_steps
                    )
                logger.debug(
                    f"len(trainer.datamodule.train_dataloader()): {len(self.trainer.datamodule.train_dataloader())}"
                )
                logger.debug(f"trainer.max_epochs: {self.trainer.max_epochs}")
                logger.debug(f"trainer.accumulate_grad_batches: {self.trainer.accumulate_grad_batches}")
        else:
            max_steps = self.trainer.max_steps

        logger.debug(f"max steps: {max_steps}")

        warmup_steps = self.hparams.warmup_steps
        if isinstance(warmup_steps, float):
            warmup_steps = int(max_steps * warmup_steps)
        print(f"warmup steps: {warmup_steps}")
        logger.debug(f"warmup steps: {warmup_steps}")
        logger.debug(f"lr_schedule: {self.hparams.lr_schedule}")
        scheduler = get_lr_scheduler(
            optimizer=optimizer,
            num_max_steps=max_steps,
            num_warmup_steps=warmup_steps,
            lr_schedule=self.hparams.lr_schedule,
            end_lr=self.hparams.end_lr,
        )

        sched = {"scheduler": scheduler, "interval": "step"}
        aug_optimizer = None
        if self.hparams.aug_optimizer and self.hparams.aug_turn_on:
            print("initilize augment optimizer")
            # augment network optimizer
            aug_grouped_parameters = get_augment_network_parameters(
                self.model, self.hparams.aug_lr
            )
            aug_optimizer = get_optimizer(
                optim_type=self.hparams.aug_optim_type,
                optimizer_grouped_parameters=aug_grouped_parameters,
                lr=self.hparams.aug_lr,
                weight_decay=self.hparams.aug_weight_decay,
            )
            return [optimizer, aug_optimizer], [sched]

        logger.debug("done configuring optimizer and scheduler")

        
        return [optimizer], [sched]

    def on_before_optimizer_step(self, optimizer):
        # If using mixed precision, the gradients are already unscaled here
        if self.track_grad_norm != -1:
            self.log_dict(grad_norm(self, norm_type=self.track_grad_norm), prog_bar=True)
