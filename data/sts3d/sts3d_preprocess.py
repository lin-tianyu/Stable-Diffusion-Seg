import os
import sys

import nibabel as nib
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import glob
import h5py
from tqdm import tqdm
import torch

from skimage import measure, morphology


def show_slice(x, y=None):
    if y is not None:
        plt.subplot(121)
        plt.imshow(x, cmap="gray")
        plt.axis("off")
        plt.colorbar()
        plt.subplot(122)
        plt.imshow(y, cmap="gray")
        plt.axis("off")
        plt.colorbar()
    else:
        plt.imshow(x, cmap="gray")
        plt.axis("off")
        plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.clf()


def minmax(x):
    return x.min(), x.max()


def set_window(x, ww=1200, wl=500):
    val_min = wl - (ww / 2)
    val_max = wl + (ww / 2)
    x_w = x.copy()
    x_w[x < val_min] = val_min
    x_w[x > val_max] = val_max
    return x_w


def normalize(x):   # -> range [0, 1]
    return (x - x.min()) / (x.max() - x.min() + 1e-8)  # * 255


def process_pipe(x):
    x = set_window(x)  # set ww/wl
    x = normalize(x)  # -> range [0, 1]
    return x


if __name__ == "__main__":

    root_path = "."
    save_path = os.path.join(root_path, "data")
    root_labelled = os.path.join(root_path, "labelled")
    stop_idx = 10e10
    labeled_slice, all_slice = 0, 0

    """ for all labeled: train+test -> slices """
    with open(os.path.join(root_path, "labeledall.txt"), "w+") as f:
        train_path = glob.glob(os.path.join(root_labelled, "image", "**"))
        for idx, path in tqdm(enumerate(train_path), desc="processing labeled data"):
            img = nib.load(path).get_fdata()
            gts = nib.load(path.replace("image", "label")).get_fdata()
            img_mask = (img != img.min())
            z = img.shape[-1]
            assert img.shape[0] == img.shape[1]
            assert len(img.shape) == 3
            assert img.shape == gts.shape

            gts = morphology.remove_small_objects(gts.astype(bool), min_size=64, connectivity=3).astype(float)
            gts *= img_mask
            img_processed = process_pipe(img)

            assert (0 <= img_processed.all() <= 1)

            for slice_idx in tqdm(range(z), desc=f"single volume"):
                img_slice = img_processed[:, :, slice_idx]
                gts_slice = gts[:, :, slice_idx]
                all_slice += 1
                if img_slice.min() == img_slice.max():  # skip empty slice
                    continue
                if gts_slice.sum() > 0:
                    labeled_slice += 1
                # print(f"[{idx:4}]", img_slice.shape, minmax(img_slice), minmax(process_pipe(img_slice)))
                # show_slice(img_slice, process_pipe(img_slice))

                save_path_h5 = os.path.join(save_path, path.split("/")[-1].split(".")[0] + f"_slice_{slice_idx}.h5")
                f.write(
                    os.path.join("data", path.split("/")[-1].split(".")[0] + f"_slice_{slice_idx}.h5")+"\n"
                )
                with h5py.File(save_path_h5, mode="w") as hf:
                    hf.create_dataset(name="image", dtype=float, data=img_slice)
                    hf.create_dataset(name="label", dtype=float, data=gts_slice)
                hf.close()

                if slice_idx == stop_idx:
                    sys.exit(0)

            if idx == stop_idx:
                sys.exit(0)
        f.close()

    """ for valtest: test -> volumes """
    with open("valtest.txt", "w+") as f:
        val_path = glob.glob(os.path.join(root_labelled, "image", "*_001*_*"))
        print(val_path)
        for idx, path in tqdm(enumerate(val_path), desc="processing unlabeled data"):
            # print(path)
            img = nib.load(path).get_fdata()
            gts = nib.load(path.replace("image", "label")).get_fdata()
            # print(img.shape, gts.shape)
            z = img.shape[-1]
            assert img.shape[0] == img.shape[1]
            assert len(img.shape) == 3
            assert img.shape == gts.shape
    
            img = torch.tensor(img).permute(2, 0, 1).numpy()
            gts = torch.tensor(gts).permute(2, 0, 1).numpy()
            # print(img.shape, gts.shape)
    
            img_processed = process_pipe(img)
            # print(minmax(img_processed), minmax(gts))
    
            save_path_h5 = os.path.join(save_path, path.split("/")[-1].split(".")[0] + ".h5")
            f.write(
                os.path.join("data", path.split("/")[-1].split(".")[0] + ".h5") + "\n"
            )
            # print(save_path_h5)
            with h5py.File(save_path_h5, mode="w") as hf:
                hf.create_dataset(name="image", dtype=float, data=img_processed)
                hf.create_dataset(name="label", dtype=float, data=gts)
            hf.close()
    
            if idx == stop_idx:
                sys.exit(0)
        f.close()


