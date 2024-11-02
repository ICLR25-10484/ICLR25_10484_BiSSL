# This code is modified version fn the original augmentation code from the VICReg implementation
# found at https://github.com/facebookresearch/vicreg/blob/main/augmentations.py

# Copyright (c) Meta Platforms, Inc. and affiliates.

from PIL import ImageOps, ImageFilter
import numpy as np
import torchvision.transforms as transforms
from timm.data import ToTensor


RBG_NORM_MEAN = [0.485, 0.456, 0.406]
RBG_NORM_STD = [0.229, 0.224, 0.225]


class GaussianBlur(object):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(radius=sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class TrainTransform(object):
    def __init__(
        self,
        img_size: int = 224,
        interpolation_mode: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        min_ratio: float = 0.08,
    ):
        self.transform = transforms.Compose(
            transforms=[
                transforms.RandomResizedCrop(
                    size=img_size,
                    interpolation=interpolation_mode,
                    antialias=True,
                    ratio=(min_ratio, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    transforms=[
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=1.0),
                # Solarization(p=0.0),
                ToTensor(),
                transforms.Normalize(mean=RBG_NORM_MEAN, std=RBG_NORM_STD),
            ]
        )
        self.transform_prime = transforms.Compose(
            transforms=[
                transforms.RandomResizedCrop(
                    size=img_size,
                    interpolation=interpolation_mode,
                    antialias=True,
                    ratio=(min_ratio, 1.0),
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    transforms=[
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                GaussianBlur(p=0.1),
                Solarization(p=0.2),
                ToTensor(),
                transforms.Normalize(mean=RBG_NORM_MEAN, std=RBG_NORM_STD),
            ]
        )

    def __call__(self, sample):
        x1 = self.transform(sample)
        x2 = self.transform_prime(sample)
        return x1, x2


class BLODownstreamTrainTransform(TrainTransform):
    def __init__(
        self,
        img_size: int = 224,
        interpolation_mode: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
        min_ratio: float = 0.08,
    ):
        super().__init__(img_size=img_size, interpolation_mode=interpolation_mode)

        self.transform_downstream = transforms.Compose(
            transforms=[
                transforms.RandomResizedCrop(
                    size=img_size,
                    ratio=(min_ratio, 1.0),
                    interpolation=interpolation_mode,
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=RBG_NORM_MEAN, std=RBG_NORM_STD),
            ]
        )

    def __call__(self, sample):
        return self.transform_downstream(sample)


class BLODownstreamTestTransform(object):
    def __init__(
        self,
        img_size: int = 224,
        interpolation_mode: transforms.InterpolationMode = transforms.InterpolationMode.BICUBIC,
    ):
        imagenet_crop_ratio: float = 224 / 256

        self.transform = transforms.Compose(
            transforms=[
                transforms.Resize(
                    size=int(img_size / imagenet_crop_ratio),
                    interpolation=interpolation_mode,
                ),
                transforms.CenterCrop(size=img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=RBG_NORM_MEAN, std=RBG_NORM_STD),
            ]
        )

    def __call__(self, sample):
        return self.transform(sample)
