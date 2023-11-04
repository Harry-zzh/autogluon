import logging
import random
import warnings
from io import BytesIO
from typing import Dict, List, Optional, Union

import PIL
import torch
from omegaconf import DictConfig
from PIL import ImageFile
from torch import nn
from torchvision import transforms

from .utils import construct_image_processor, image_mean_std

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = PIL.Image.BICUBIC

from ..constants import COLUMN, IMAGE, IMAGE_BYTEARRAY, IMAGE_VALID_NUM, LABEL
from .collator import PadCollator, StackCollator

logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RealWorldSemSegImageProcessor:
    """
    Prepare image data for the model specified by "prefix". For multiple models requiring image data,
    we need to create a ImageProcessor for each related model so that they will have independent input.
    """

    def __init__(
        self,
        model: nn.Module,
        model_config: DictConfig,
        norm_type: Optional[str] = None,
        max_img_num_per_col: Optional[int] = 1,
        missing_value_strategy: Optional[str] = "zero",
        requires_column_info: bool = False,
    ):
        """
        Parameters
        ----------
        model
            The model for which this processor would be created.
        model_config
            The config of the model.
        norm_type
            How to normalize an image. We now support:
            - inception
                Normalize image by IMAGENET_INCEPTION_MEAN and IMAGENET_INCEPTION_STD from timm
            - imagenet
                Normalize image by IMAGENET_DEFAULT_MEAN and IMAGENET_DEFAULT_STD from timm
            - clip
                Normalize image by mean (0.48145466, 0.4578275, 0.40821073) and
                std (0.26862954, 0.26130258, 0.27577711), used for CLIP.
        max_img_num_per_col
            The maximum number of images one sample can have.
        missing_value_strategy
            How to deal with a missing image. We now support:
            - skip
                Skip this sample
            -zero
                Use an image with zero pixels.
        requires_column_info
            Whether to require feature column information in dataloader.
        """

        self.model_config = model_config

        train_transforms, val_transforms = self.get_img_transform(kind="img")
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        logger.debug(f"image training transforms: {self.train_transforms}")
        logger.debug(f"image validation transforms: {self.val_transforms}")

        gt_train_transforms, gt_val_transforms = self.get_img_transform(kind="gt")
        self.gt_train_transforms = gt_train_transforms
        self.gt_val_transforms = gt_val_transforms
        logger.debug(f"gt training transforms: {self.gt_train_transforms}")
        logger.debug(f"gt validation transforms: {self.gt_val_transforms}")

        self.prefix = model.prefix
        self.missing_value_strategy = missing_value_strategy
        self.requires_column_info = requires_column_info

        self.size = model.image_size
        self.mean, self.std = image_mean_std(model.config["image_norm"])

        self.normalization = transforms.Normalize(self.mean, self.std)

        self.max_img_num_per_col = max_img_num_per_col
        if max_img_num_per_col <= 0:
            logger.debug(f"max_img_num_per_col {max_img_num_per_col} is reset to 1")
            max_img_num_per_col = 1
        self.max_img_num_per_col = max_img_num_per_col
        logger.debug(f"max_img_num_per_col: {max_img_num_per_col}")

        self.train_processor = construct_image_processor(
            image_transforms=self.train_transforms, size=self.size, normalization=self.normalization
        )
        self.val_processor = construct_image_processor(
            image_transforms=self.val_transforms, size=self.size, normalization=self.normalization
        )

        self.gt_train_processor = construct_image_processor(
            image_transforms=self.gt_train_transforms, size=self.size, normalization=None
        )
        self.gt_val_processor = construct_image_processor(
            image_transforms=self.gt_val_transforms, size=self.size, normalization=None
        )
        self.augmentation = self.get_aug_transform()

    @property
    def image_key(self):
        return f"{self.prefix}_{IMAGE}"

    @property
    def label_key(self):
        return f"{self.prefix}_{LABEL}"

    @property
    def image_valid_num_key(self):
        return f"{self.prefix}_{IMAGE_VALID_NUM}"

    @property
    def image_column_prefix(self):
        return f"{self.image_key}_{COLUMN}"

    def collate_fn(self, image_column_names: Optional[List] = None, per_gpu_batch_size: Optional[int] = None) -> Dict:
        """
        Collate images into a batch. Here it pads images since the image number may
        vary from sample to sample. Samples with less images will be padded zeros.
        The valid image numbers of samples will be stacked into a vector.
        This function will be used when creating Pytorch DataLoader.

        Returns
        -------
        A dictionary containing one model's collator function for image data.
        """
        fn = {}
        if self.requires_column_info:
            assert image_column_names, "Empty image column names."
            for col_name in image_column_names:
                fn[f"{self.image_column_prefix}_{col_name}"] = StackCollator()

        fn.update(
            {
                self.image_key: PadCollator(pad_val=0),
                self.image_valid_num_key: StackCollator(),
                self.label_key: PadCollator(pad_val=0),
            }
        )

        return fn

    def process_one_sample(
        self,
        image_features: Dict[str, Union[List[str], List[bytearray]]],
        feature_modalities: Dict[str, List[str]],
        is_training: bool,
        image_mode: Optional[str] = "RGB",
    ) -> Dict:
        """
        Read images, process them, and stack them. One sample can have multiple images,
        resulting in a tensor of (n, 3, size, size), where n <= max_img_num_per_col is the available image number.

        Parameters
        ----------
        image_features
            One sample may have multiple image columns in a pd.DataFrame and multiple images
            inside each image column.
        feature_modalities
            What modality each column belongs to.
        is_training
            Whether to process images in the training mode.
        image_mode
            A string which defines the type and depth of a pixel in the image.
            For example, RGB, RGBA, CMYK, and etc.

        Returns
        -------
        A dictionary containing one sample's images and their number.
        """
        images = []
        gts = []
        zero_images = []
        ret = {}

        per_col_image_features, per_col_gt_features = (image_features["image"], image_features["gt"])
        for img_feature, gt_feature in zip(
            per_col_image_features[: self.max_img_num_per_col], per_col_gt_features[: self.max_img_num_per_col]
        ):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=("Palette images with Transparency expressed in bytes should be converted to RGBA images"),
                )
                try:
                    if feature_modalities.get("image") == IMAGE_BYTEARRAY:
                        image_feature = BytesIO(img_feature)
                    else:
                        image_feature = img_feature
                    with PIL.Image.open(image_feature) as img:
                        img = img.convert(image_mode)
                except Exception as e:
                    if self.missing_value_strategy.lower() == "zero":
                        logger.debug(f"Using a zero image due to '{e}'")
                        img = PIL.Image.new(image_mode, (self.size, self.size), color=0)
                    else:
                        raise e

                with PIL.Image.open(gt_feature) as gt:
                    gt = gt.convert("L")

            if is_training:
                if random.random() < 0.5:
                    img = self.augmentation(img)
                    gt = self.augmentation(gt)
                img = self.train_processor(img)
                gt = self.gt_train_processor(gt)
            else:
                img = self.val_processor(img)
                gt = self.gt_val_processor(gt)

            images.append(img)
            gts.append(gt)

        ret.update(
            {
                self.image_key: torch.tensor([])
                if len(images + zero_images) == 0
                else torch.cat(images + zero_images, dim=0),
                self.image_valid_num_key: len(images),
                self.label_key: torch.cat(gts, dim=0),
            }
        )
        return ret

    def __call__(
        self,
        images: Dict[str, List[str]],
        feature_modalities: Dict[str, Union[int, float, list]],
        is_training: bool,
    ) -> Dict:
        """
        Obtain one sample's images and customized them for a specific model.

        Parameters
        ----------
        images
            Images of one sample.
        feature_modalities
            The modality of the feature columns.
        is_training
            Whether to process images in the training mode.

        Returns
        -------
        A dictionary containing one sample's processed images and their number.
        """
        images = {k: [v] if isinstance(v, str) else v for k, v in images.items()}

        return self.process_one_sample(images, feature_modalities, is_training)

    def __getstate__(self):
        odict = self.__dict__.copy()  # get attribute dictionary
        del odict["train_processor"]  # remove augmenter to support pickle
        return odict

    def __setstate__(self, state):
        self.__dict__ = state
        if "train_transform_types" in state:  # backward compatible
            self.train_transforms = list(self.train_transform_types)
        if "val_transform_types" in state:
            self.val_transforms = list(self.val_transform_types)

        self.train_processor = construct_image_processor(
            image_transforms=self.train_transforms,
            size=self.size,
            normalization=self.normalization,
        )

    def get_img_transform(self, kind: str):
        if kind == "img":
            train_transforms = self.model_config.image.train_transforms
            val_transforms = self.model_config.image.val_transforms
        elif kind == "gt":
            train_transforms = self.model_config.gt.train_transforms
            val_transforms = self.model_config.gt.val_transforms
        else:
            raise ValueError("Do not support the usage!")
        train_transforms = list(train_transforms)
        val_transforms = list(val_transforms)

        return train_transforms, val_transforms

    def get_aug_transform(self):
        aug_transforms = self.model_config.aug
        aug_transforms = list(aug_transforms)
        aug_trans = []
        for trans_mode in aug_transforms:
            if trans_mode == "random_horizontal_flip":
                aug_trans.append(transforms.RandomHorizontalFlip(1.0))
        return transforms.Compose(aug_trans)