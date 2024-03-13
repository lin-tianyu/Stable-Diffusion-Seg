import os
import sys

import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage.interpolation import zoom

import glob
import h5py

ROOT_PATH = "data/sts3d/"

class STS3DBase(Dataset):
    """STS-3D Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None):
        self.data_root = data_root
        # self.data_paths = glob.glob(os.path.join(self.data_root, "*.png"))  # seg map, img slice with *.npy postfix

        self.mode = mode
        assert mode in ["train", "val", "test"]
        if mode == "train":
            with open(os.path.join(ROOT_PATH, "splits", "labeled.txt"), 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == "val":
            with open(os.path.join(ROOT_PATH, "splits", "val.txt"), 'r') as f:
                self.ids = f.read().splitlines()
        elif mode == "test":
            with open(os.path.join(ROOT_PATH, "splits", "valtest.txt"), 'r') as f:
                self.ids = f.read().splitlines()
        else:
            raise NotImplementedError
        self._length = len(self.ids)

        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop(size=(256, 256)),
        ])
        print(f"[Dataset]: STS-3D with 2 classes, in {self.mode} mode")

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        # read segmentation and images
        data_path = self.ids[i]
        example = dict(file_path_=data_path)

        if self.mode == "test":
            sample = h5py.File(os.path.join(self.data_root, data_path), "r")
            image = sample["image"][:].transpose([1, 2, 0])
            segmentation = sample["label"][:].transpose([1, 2, 0])
            example["image"] = (image * 2) - 1
            example["segmentation"] = segmentation
            return example

        sample = h5py.File(os.path.join(self.data_root, data_path), "r")
        image = sample["image"][:]
        try:
            segmentation = sample["label"][:]
        except:
            segmentation = np.zeros_like(image)
        x, y = image.shape
        image = zoom(image, (self.size / x, self.size / y), order=1)    # order=1 for bilinear interpolation
        segmentation = zoom(segmentation, (self.size / x, self.size / y), order=0)  # order=0 for nearest interpolation
        segmentation = torch.tensor(segmentation).repeat(3, 1, 1)
        image = torch.tensor(image).repeat(3, 1, 1)
        # print(segmentation.shape, image.shape)

        if self.mode == "train":
            state = torch.get_rng_state()
            segmentation = self.transform(segmentation)
            torch.set_rng_state(state)
            image = self.transform(image)

        # only support binary segmentation now:
        segmentation = np.array(segmentation.permute(1, 2, 0))
        image = np.array(image.permute(1, 2, 0))
        segmentation = np.where(segmentation > 0.5, 1, 0)   
        example["segmentation"] = ((segmentation * 2) - 1).astype(np.float32)   # range: binary -1 and 1
        example["image"] = ((image * 2) - 1).astype(np.float32)     # range from -1 to 1, np.float32
        return example


class STS3DTrain(STS3DBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/sts3d/", mode="train", **kwargs)


class STS3DValidation(STS3DBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/sts3d/", mode="val", **kwargs)

class STS3DTest(STS3DBase):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/sts3d/", mode="test", **kwargs)


# class STS3DNoEmptyTrain(STS3DBase):
#     def __init__(self, **kwargs):
#         super().__init__(data_root=DATA_Path, mode="train", **kwargs)



