import os
import glob
import h5py
import re
import imageio
# Building an H5 Dataset
# DF2K includes DIV2K and Flickr2K
dataset_dir = "./your_dir/DF2K"
hr_paths = glob.glob(os.path.join(dataset_dir, "DF2K_train_HR", "*.png"))
# You need to build H5 training data for X2, X3, X4 Scale, which is an example of X2
lr_paths = glob.glob(os.path.join(dataset_dir, "DF2K_train_LR_bicubic", "X2", "*.png"))
scale = 2

def make_hr_map(paths):
    out = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        out[name] = p
    return out

def make_lr_map(paths):
    out = {}
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        name = re.sub(r'([_-]?[xX]2)$', '', name)
        out[name] = p
    return out

hr_map = make_hr_map(hr_paths)
lr_map = make_lr_map(lr_paths)
common_keys = sorted(set(hr_map.keys()) & set(lr_map.keys()))
print("HR:", len(hr_map))
print("LR:", len(lr_map))
print("Common:", len(common_keys))
valid = 0
bad = 0

with h5py.File("DF2K_x2_fixed.h5", "w") as f:
    grp_hr = f.create_group("HR")
    grp_lr = f.create_group("X2")
    total = len(common_keys)
    for i, key in enumerate(common_keys):
        if i % 100 == 0:
            print(f"[{i}/{total}] Processing {key}")
        hr = imageio.imread(hr_map[key])
        lr = imageio.imread(lr_map[key])
        if hr.shape[0] != lr.shape[0] * scale or hr.shape[1] != lr.shape[1] * scale:
            print(f"[BAD PAIR] {key}: HR={hr.shape}, LR={lr.shape}")
            bad += 1
            continue
        grp_hr.create_dataset(str(valid), data=hr)
        grp_lr.create_dataset(str(valid), data=lr)
        valid += 1

print("Valid pairs:", valid)
print("Bad pairs:", bad)