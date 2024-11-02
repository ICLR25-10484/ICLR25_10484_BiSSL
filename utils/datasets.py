from pathlib import Path
from typing import Literal, Any, List, Optional, Callable
from torch.utils.data import Dataset

import numpy as np
from torchvision import datasets
from torchvision.transforms import Compose, Grayscale

from utils.dset_attr.classes import flowers_classes

from utils.dset_attr.modified_datasets import (
    Caltech101GreyToRGB,
    SUN397WPartitions,
    VOC2007Classification,
    CUB2011,
)


def get_subset_samples_trainval(
    data: Dataset,
    data_idxs: List[int],
    return_separate_data_labels: bool = False,
):

    samples_subset = [data[i] for i in data_idxs]

    if return_separate_data_labels:
        samps, labels = zip(*samples_subset)
        samps, labels = np.array(samps), np.array(labels)
        return samps, labels

    return samples_subset


class GetData:
    def __init__(self, root: str, download: bool = False):
        self.root: str | Path = root
        self.download: bool = download

    def __call__(
        self,
        dset_name: Literal[
            "stl10",
            "food",
            "fashionmnist",
            "cars",
            "dtd",
            "pets",
            "flowers",
            "aircrafts",
            "cifar10",
            "cifar100",
            "caltech101",
            "sun397",
            "voc2007",
            "cub2011",
        ],
        split: Literal["train", "val", "test", "unlabeled", "trainval"],
        transform: Optional[Callable],
    ) -> (
        datasets.STL10
        | datasets.Food101
        | datasets.FashionMNIST
        | datasets.StanfordCars
        | datasets.DTD
        | datasets.OxfordIIITPet
        | datasets.Flowers102
        | datasets.FGVCAircraft
        | datasets.CIFAR10
        | datasets.CIFAR100
        | Caltech101GreyToRGB
        | SUN397WPartitions
        | VOC2007Classification
        | CUB2011
    ):
        match dset_name:
            case "stl10":
                assert split in ["unlabeled", "train", "val", "test"]
                dset = datasets.STL10(
                    root=self.root,
                    split=split if split != "val" else "train",
                    download=self.download,
                    transform=transform,
                )
                if split in ["train", "val"]:
                    dset_idxs: np.ndarray[Any, np.dtype[Any]] = np.loadtxt(
                        fname=self.root
                        + "/tv_splits/"
                        + dset_name
                        + "_"
                        + split
                        + "_split.txt",
                        dtype=int,
                    )
                    dset.data, dset.labels = get_subset_samples_trainval(
                        data=list(zip(dset.data, dset.labels)),
                        data_idxs=dset_idxs,
                        return_separate_data_labels=True,
                    )
                return dset
            case "food":
                assert split in ["train", "val", "test"]

                dset = datasets.Food101(
                    root=self.root,
                    split=split if split != "val" else "train",
                    download=self.download,
                    transform=transform,
                )
                if split in ["train", "val"]:
                    dset_idxs: np.ndarray[Any, np.dtype[Any]] = np.loadtxt(
                        fname=self.root
                        + "/tv_splits/"
                        + dset_name
                        + "_"
                        + split
                        + "_split.txt",
                        dtype=int,
                    )
                    dset._image_files, dset._labels = get_subset_samples_trainval(
                        data=list(zip(dset._image_files, dset._labels)),
                        data_idxs=dset_idxs,
                        return_separate_data_labels=True,
                    )

                return dset

            case "fashionmnist":
                assert split in ["train", "val", "test"]
                train = True if split in ["train", "val"] else False

                dset = datasets.FashionMNIST(
                    root=self.root,
                    train=train,
                    download=self.download,
                    transform=Compose([Grayscale(num_output_channels=3), transform]),
                )
                if split in ["train", "val"]:
                    dset_idxs = np.loadtxt(
                        self.root
                        + "/tv_splits/"
                        + dset_name
                        + "_"
                        + split
                        + "_split.txt",
                        dtype=int,
                    )
                    dset.data, dset.targets = get_subset_samples_trainval(
                        data=list(zip(dset.data, dset.targets)),
                        data_idxs=dset_idxs,
                        return_separate_data_labels=True,
                    )

                return dset

            case "cars":
                assert split in ["train", "val", "test"]
                dset = datasets.StanfordCars(
                    root=self.root,
                    split=split if split != "val" else "train",
                    download=self.download,
                    transform=transform,
                )
                if split in ["train", "val"]:
                    dset_idxs = np.loadtxt(
                        self.root
                        + "/tv_splits/"
                        + dset_name
                        + "_"
                        + split
                        + "_split.txt",
                        dtype=int,
                    )
                    dset._samples = list(
                        get_subset_samples_trainval(
                            data=dset._samples,
                            data_idxs=dset_idxs,
                        )
                    )
                return dset

            case "dtd":
                assert split in ["train", "test", "val"]
                dset = datasets.DTD(
                    root=self.root,
                    split=split,
                    download=self.download,
                    transform=transform,
                )
                return dset

            case "pets":
                assert split in ["train", "val", "test"]
                split_get = "trainval" if split in ["train", "val"] else "test"
                dset = datasets.OxfordIIITPet(
                    root=self.root,
                    split=split_get,
                    download=self.download,
                    transform=transform,
                )

                if split in ["train", "val"]:
                    dset_idxs = np.loadtxt(
                        self.root
                        + "/tv_splits/"
                        + dset_name
                        + "_"
                        + split
                        + "_split.txt",
                        dtype=int,
                    )
                    dset._images, dset._labels = get_subset_samples_trainval(
                        data=list(zip(dset._images, dset._labels)),
                        data_idxs=dset_idxs,
                        return_separate_data_labels=True,
                    )
                return dset

            case "flowers":
                assert split in ["train", "val", "test"]
                dset = datasets.Flowers102(
                    root=self.root,
                    split=split,
                    download=self.download,
                    transform=transform,
                )
                dset.classes = flowers_classes

                return dset

            case "aircrafts":
                assert split in ["train", "val", "trainval", "test"]
                dset = datasets.FGVCAircraft(
                    root=self.root,
                    split=split,
                    download=self.download,
                    transform=transform,
                )
                return dset

            case "cifar10":
                assert split in ["train", "val", "test"]
                train = True if split in ["train", "val"] else False

                dset = datasets.CIFAR10(
                    root=self.root,
                    train=train,
                    download=self.download,
                    transform=transform,
                )

                if split in ["train", "val"]:
                    dset_idxs = np.loadtxt(
                        self.root
                        + "/tv_splits/"
                        + dset_name
                        + "_"
                        + split
                        + "_split.txt",
                        dtype=int,
                    )
                    dset.data, dset.targets = get_subset_samples_trainval(
                        data=list(zip(dset.data, dset.targets)),
                        data_idxs=dset_idxs,
                        return_separate_data_labels=True,
                    )

                return dset

            case "cifar100":
                assert split in ["train", "val", "test"]
                train = True if split in ["train", "val"] else False

                dset = datasets.CIFAR100(
                    root=self.root,
                    train=train,
                    download=self.download,
                    transform=transform,
                )

                if split in ["train", "val"]:
                    dset_idxs = np.loadtxt(
                        self.root
                        + "/tv_splits/"
                        + dset_name
                        + "_"
                        + split
                        + "_split.txt",
                        dtype=int,
                    )
                    dset.data, dset.targets = get_subset_samples_trainval(
                        data=list(zip(dset.data, dset.targets)),
                        data_idxs=dset_idxs,
                        return_separate_data_labels=True,
                    )

                return dset

            case "caltech101":
                assert split in ["train", "val", "test"]
                dset = Caltech101GreyToRGB(
                    root=self.root,
                    download=self.download,
                    transform=transform,
                )
                dset.classes = dset.categories

                dset_idxs = np.loadtxt(
                    self.root + "/tv_splits/" + dset_name + "_" + split + "_split.txt",
                    dtype=int,
                )
                dset.index, dset.y = get_subset_samples_trainval(
                    data=list(zip(dset.index, dset.y)),
                    data_idxs=dset_idxs,
                    return_separate_data_labels=True,
                )
                return dset

            case "sun397":
                assert split in ["train", "test", "val"]
                dset = SUN397WPartitions(
                    root=self.root,
                    split=split if split != "val" else "train",
                    download=self.download,
                    transform=transform,
                )
                if split in ["train", "val"]:
                    dset_idxs = np.loadtxt(
                        self.root
                        + "/tv_splits/"
                        + dset_name
                        + "_"
                        + split
                        + "_split.txt",
                        dtype=int,
                    )
                    dset._image_files, dset._labels = get_subset_samples_trainval(
                        data=list(zip(dset._image_files, dset._labels)),
                        data_idxs=dset_idxs,
                        return_separate_data_labels=True,
                    )

                return dset

            case "voc2007":
                assert split in ["train", "test", "val"]
                dset = VOC2007Classification(
                    root=self.root,
                    image_set=split,
                    download=self.download,
                    transform=transform,
                )

                return dset

            case "cub2011":
                assert split in ["train", "test", "val"]
                dset = CUB2011(
                    root=self.root,
                    train=False if split == "test" else True,
                    download=self.download,
                    transform=transform,
                )
                if split in ["train", "val"]:
                    dset_idxs = np.loadtxt(
                        self.root
                        + "/tv_splits/"
                        + dset_name
                        + "_"
                        + split
                        + "_split.txt",
                        dtype=int,
                    )
                    dset.images, dset.targets = get_subset_samples_trainval(
                        data=list(zip(dset.images, dset.targets)),
                        data_idxs=dset_idxs,
                        return_separate_data_labels=True,
                    )
                return dset

            case _:
                raise ValueError(f'"{dset_name}" is an invalid dataset.')
