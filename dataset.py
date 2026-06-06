from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement
import os
import glob
import h5py
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def random_crop(hr, lr, size, scale):
    h, w = lr.shape[:2]

    if h < size or w < size:
        raise ValueError(
            f"LR patch too small for crop: lr shape={lr.shape}, required size={size}"
        )

    h_hr, w_hr = hr.shape[:2]
    hsize = size * scale

    if h_hr < hsize or w_hr < hsize:
        raise ValueError(
            f"HR patch too small for crop: hr shape={hr.shape}, required size={hsize}"
        )

    x = random.randint(0, w - size)
    y = random.randint(0, h - size)

    hx, hy = x * scale, y * scale

    crop_lr = lr[y:y + size, x:x + size].copy()
    crop_hr = hr[hy:hy + hsize, hx:hx + hsize].copy()

    if crop_lr.shape[:2] != (size, size):
        raise ValueError(
            f"Bad LR crop shape: got {crop_lr.shape}, expected {(size, size)}"
        )
    if crop_hr.shape[:2] != (hsize, hsize):
        raise ValueError(
            f"Bad HR crop shape: got {crop_hr.shape}, expected {(hsize, hsize)}"
        )

    return crop_hr, crop_lr


def random_flip_and_rotate(im1, im2):
    if random.random() < 0.5:
        im1 = np.flipud(im1)
        im2 = np.flipud(im2)

    if random.random() < 0.5:
        im1 = np.fliplr(im1)
        im2 = np.fliplr(im2)

    angle = random.choice([0, 1, 2, 3])
    im1 = np.rot90(im1, angle)
    im2 = np.rot90(im2, angle)

    # have to copy before be called by transform function
    return im1.copy(), im2.copy()


class TrainDataset(data.Dataset):
    def __init__(self, path, size, scale):
        super(TrainDataset, self).__init__()

        self.size = size
        h5f = h5py.File(path, "r")

        self.hr = [v[:] for v in h5f["HR"].values()]

        self.scale = [scale]
        self.lr = [[v[:] for v in h5f["X{}".format(scale)].values()]]

        h5f.close()

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        size = self.size

        item = [(self.hr[index], self.lr[i][index]) for i, _ in enumerate(self.lr)]
        item = [random_crop(hr, lr, size, self.scale[i]) for i, (hr, lr) in enumerate(item)]
        item = [random_flip_and_rotate(hr, lr) for hr, lr in item]

        return [(self.transform(hr), self.transform(lr)) for hr, lr in item]

    def __len__(self):
        return len(self.hr)


class TestDataset(data.Dataset):
    def __init__(self, dirname, scale, hr_subdir="HR", lr_subdir_pattern="LR_bicubic/X{}"):
        super(TestDataset, self).__init__()
        self.name = dirname.split("/")[-1]
        abs_dirname = os.path.abspath(dirname)
        self.scale = scale
        # 打印调试信息
        print("-" * 40)
        # print(f"[Debug TestDataset INIT] Initializing TestDataset...")
        # print(f"[Debug TestDataset INIT] Received base directory (dirname): '{dirname}'")
        # print(f"[Debug TestDataset INIT] Absolute base directory: '{abs_dirname}'")
        # print(f"[Debug TestDataset INIT] Scale: {scale}")
        # print(f"[Debug TestDataset INIT] HR subdirectory pattern: '{hr_subdir}'")
        # print(f"[Debug TestDataset INIT] LR subdirectory pattern: '{lr_subdir_pattern}'")
        hr_folder = os.path.join(abs_dirname, hr_subdir)
        try:
            lr_folder = os.path.join(abs_dirname, lr_subdir_pattern.format(scale))
        except Exception as e:
            # 如果 lr_subdir_pattern 不包含 '{}' 或格式错误
            # print(f"[Error TestDataset INIT] Exception during LR folder path creation: {e}")
            self.hr = []  # 设置为空列表
            self.lr = []  # 设置为空列表
            # 可以选择直接使用原始模式，或抛出错误
            lr_folder = os.path.join(abs_dirname, lr_subdir_pattern)
        # print(f"[Debug TestDataset INIT] Searching for HR images (*.png) in: '{hr_folder}'")
        # print(f"[Debug TestDataset INIT] Searching for LR images (*.png) in: '{lr_folder}'")
        # 使用 glob 查找所有 png 文件
        self.hr = sorted(glob.glob(os.path.join(hr_folder, "*.png")))
        self.lr = sorted(glob.glob(os.path.join(lr_folder, "*.png")))
        # 打印查找结果
        print(f"[Debug TestDataset INIT] Found {len(self.hr)} HR image paths.")
        if not self.hr:
            print(f"  [Warning] No HR images found. Check the path and '{hr_subdir}' subdirectory.")
        # else:
            # print(f"  Example HR paths: {self.hr[:min(3, len(self.hr))]}")  # 打印前几个示例

        print(f"[Debug TestDataset INIT] Found {len(self.lr)} LR image paths.")
        if not self.lr:
            print(
                    f"  [Warning] No LR images found for scale {scale}. Check the path and '{lr_subdir_pattern.format(scale)}' subdirectory.")
        # else:
        #     print(f"  Example LR paths: {self.lr[:min(3, len(self.lr))]}")  # 打印前几个示例
            # 可选的健壮性检查：确保 HR 和 LR 数量匹配
            # if len(self.hr) != len(self.lr):
            #     print(
            #             f"[Warning TestDataset INIT] Mismatch in number of HR ({len(self.hr)}) and LR ({len(self.lr)}) images found. "
            #             "This might lead to errors during evaluation or incorrect pairings.")
            print("-" * 40)
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        hr = Image.open(self.hr[index])
        lr = Image.open(self.lr[index])

        hr = hr.convert("RGB")
        lr = lr.convert("RGB")
        filename = self.hr[index].split("/")[-1]

        return self.transform(hr), self.transform(lr), filename

    def __len__(self):
        return len(self.hr)
