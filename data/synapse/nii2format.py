"""
Transforming BTCV Dataset into .npz format for Dataloader.
Also, generating .txt files that contain .npz file name.
- Train set .npz: 'image' data and 'label' data pair
- Test  set .npz: 'image' data
"""

import os
import glob
from tqdm import tqdm
import nibabel as nib
import numpy as np
import pandas as pd
import cv2
from pprint import pprint
from PIL import Image



TEST_VOLUME = [
    "0001",
    "0002",
    "0003",
    "0004",
    "0008",
    "0022",
    "0025",
    "0029",
    "0032",
    "0035",
    "0036",
    "0038",
]  # 12 testing volume, following `TransUnet`, etc.



def generate_vol_txt():
    dataset = "Abdomen"
    imgpath = sorted(glob.glob(os.path.join(root_path, dataset, 'RawData/Training/img', '*.nii.gz')))

    test_imgpath = list(filter(lambda x:
                               x if "".join(list(filter(str.isdigit, x.split('/')[-1].split('.')[0]))) in TEST_VOLUME
                               else False, imgpath))

    return test_imgpath


def label_range_detector(gts):
    # gts : 512, 512, z
    x_list, y_list = list(), list()
    for slice in range(gts.shape[0]):
        if gts[slice, :, :].sum() > 0:
            x_list.append(slice)
    for slice in range(gts.shape[1]):
        if gts[:, slice, :].sum() > 0:
            y_list.append(slice)
    return x_list[0], gts.shape[0] - x_list[-1], y_list[0], gts.shape[1] - y_list[-1]


def generate_npz(img_path, test_vol_path, mode="all"):
    assert mode in ["labeled_only", "all"], "expect 'labeled_only' or 'all' ! "
    """generating npz files and txt files"""
    train_image_path = list(set(img_path) - set(test_vol_path))
    train_label_path = list(map(lambda x: x.replace("img", "label"), train_image_path))
    test_image_path = test_vol_path
    test_label_path = list(map(lambda x: x.replace("img", "label"), test_image_path))

    # opertaing dataset and store in ddpm-segmentation format
    def operate_data(image_path, label_path, train_set: bool):
        if train_set:
            train_set = "train"
        else:
            train_set = "test"

        if not os.path.exists(train_set):
            os.mkdir(train_set)

        min_edge = list()
        slice_counter = 0
        pixels_count = np.zeros(14)
        for imgpath, gtspath in tqdm(zip(image_path, label_path),
                                     desc=f'processing {dataset} ({train_set})',
                                     total=len(image_path)):
            img = nib.load(imgpath).get_fdata()
            gts = nib.load(gtspath).get_fdata()
            x1, x2, y1, y2 = label_range_detector(gts)  # detect label range
            min_edge.append(min([x1, x2, y1, y2]))
            assert img.shape == gts.shape
            img[img < -175] = -175  # set WW&WL
            img[img > 250] = 250
            for slice in range(img.shape[2]):
                # if train_set == 'train' and gts[:, :, slice].astype(np.uint8).sum() == 0:
                #     continue
                slice_counter += 1
                # img preprocess
                slice_img = img[:, :, slice].astype(np.float32)
                slice_img = cv2.resize(slice_img, (256, 256))  # from 512x512 -> 256x256
                slice_img = np.repeat(slice_img[:, :, np.newaxis], 3, axis=2)  # shape: 256, 256, 3
                slice_img = (((slice_img + 125.) / 400.) * 2.) - 1.  # convert to range [-1, 1]

                # gts preprocess
                slice_gts = gts[:, :, slice].astype(np.uint8)
                slice_gts = cv2.resize(slice_gts, (256, 256),
                                       interpolation=cv2.INTER_NEAREST)  # from 512x512 -> 256x256
                slice_gts = np.repeat(slice_gts[:, :, np.newaxis], 3, axis=2)  # shape: 256, 256, 3

                # count each class
                for cls in range(14):
                    pixels_count[cls] += (slice_gts[:, :, 0] == cls).sum()

                # save paths
                save_img = os.path.join(
                    save_path[train_set],
                    imgpath.split('/')[-1].split('.')[0].replace('img', 'vol') + f"_{slice}.npy")
                save_gts = os.path.join(
                    save_path[train_set],
                    imgpath.split('/')[-1].split('.')[0].replace('img', 'vol') + f"_{slice}.png")
                
                # saving
                np.save(save_img, slice_img)
                Image.fromarray(slice_gts).save(save_gts)

        print(min_edge, "minimum edge for %s :" % dataset, min(min_edge))
        print(f"[slices for {train_set} set]: {slice_counter}")

        print(pixels_count)
        print(pixels_count / pixels_count.sum())

    operate_data(train_image_path, train_label_path, train_set=True)
    operate_data(test_image_path, test_label_path, train_set=False)

if __name__ == '__main__':
    dataset = 'Abdomen'

    info = pd.DataFrame(columns=['max', 'min', 'mean', 'std', 'var', 'shape', 'dtype', 'type', 'grad', 'is_cuda',
                                 'set', 'datatype'],
                        index=None)

    root_path = "."
    save_root = "."

    # ../dataset: .nii.gz files
    # test_path = os.path.join(root_path, 'dataset', dataset, 'Rawdata', 'Testing', 'img')
    train_path = dict(
        image=os.path.join(root_path, dataset, 'RawData', 'Training', 'img'),
        label=os.path.join(root_path, dataset, 'RawData', 'Training', 'label')
    )

    # ../data: .npz files
    save_path = dict(
        train=os.path.join(save_root, 'train'),
        test=os.path.join(save_root, 'test')
    )

    train_image_path = sorted(glob.glob(train_path['image'] + '/*.nii.gz'))
    train_label_path = sorted(glob.glob(train_path['label'] + '/*.nii.gz'))
    # test_image_path = sorted(glob.glob(test_path + '/*.nii.gz'))

    test_vol_path = generate_vol_txt()
    # print(train_image_path)
    # print(test_vol_path)
    generate_npz(train_image_path, test_vol_path, mode="all")
