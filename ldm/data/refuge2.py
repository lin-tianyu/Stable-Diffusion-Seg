import os
import numpy as np
import PIL
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import glob


class REFUGE2Base(Dataset):
    """REFUGE2 Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, data_root, size=256, interpolation="nearest", mode=None, num_classes=2):
        self.data_root = data_root
        self.mode = mode
        assert mode in ["train", "val", "test"]
        self.data_paths = self._parse_data_list()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=[path for path in self.data_paths])
        self.size = size
        self.interpolation = dict(nearest=PIL.Image.NEAREST)[interpolation]   # for segmentation slice
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            # transforms.CenterCrop(size=(256, 256))
        ])
        # TODO: more data transformation

        print(f"[Dataset]: REFUGE-2 with 2 classes, in {self.mode} mode")

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)
        segmentation = Image.open(example["file_path_"].replace("images", "masks")).convert("RGB")
        image = Image.open(example["file_path_"]).convert("RGB")    # same name, different postfix

        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=PIL.Image.NEAREST)
            image = image.resize((self.size, self.size), resample=PIL.Image.BILINEAR)

        if self.mode == "train":
            segmentation, image = self._utilize_transformation(segmentation, image, self.transform)

        # only support binary segmentation now:
        segmentation = np.array(segmentation).astype(np.float32)
        cup_label = (segmentation[:, :, 1] == 255.)         # extract `cup` segmentation
        segmentation = np.zeros((self.size, self.size, 3))
        segmentation[cup_label] = 1.
        if self.mode == "test":
            example["segmentation"] = segmentation   
        else:
            example["segmentation"] = ((segmentation * 2) - 1)   # range: binary -1 and 1

        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.                            # range from -1 to 1, np.float32
        example["image"] = image
        example["class_id"] = np.array([-1])  # doesn't matter for binary seg

        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.
        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self):
        all_imgs = glob.glob(os.path.join(self.data_root, "*.png"))
        val_imgs = glob.glob(os.path.join(self.data_root, "V*.png"))
        test_imgs = glob.glob(os.path.join(self.data_root, "T*.png"))
        train_imgs = list(set(all_imgs) - set(test_imgs))  # - set(val_imgs)
        assert len(train_imgs) == 800 and len(test_imgs) == 400

        if self.mode == "train":
            return train_imgs
        elif self.mode == "val":
            return val_imgs
        elif self.mode == "test":
            return test_imgs
        else:
            raise NotImplementedError(f"Only support dataset split: train, val, test !")

    @staticmethod
    def _utilize_transformation(segmentation, image, func):
        state = torch.get_rng_state()
        segmentation = func(segmentation)
        torch.set_rng_state(state)
        image = func(image)
        return segmentation, image


class REFUGE2Train(REFUGE2Base):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/refuge2/Train_crop/images", mode="train", **kwargs)


class REFUGE2Validation(REFUGE2Base):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/refuge2/Train_crop/images", mode="test", **kwargs)


class REFUGE2Test(REFUGE2Base):
    def __init__(self, **kwargs):
        super().__init__(data_root="data/refuge2/Train_crop/images", mode="test", **kwargs)
