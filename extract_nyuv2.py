import argparse
import gzip
import os
import pickle
import shutil
import zipfile

from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import matplotlib.pyplot as plt
import numpy as np

from scipy.io import loadmat
from skimage import io
from tqdm import tqdm


def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap


def extract_images(imgs, splits, IMAGE_DIR):
    print("Extracting images...")
    imgs = imgs.transpose(0, 3, 2, 1)
    for s in ['train', 'test']:
        os.makedirs(os.path.join(IMAGE_DIR, s), exist_ok=True)
        idxs = splits[s+'Ndxs'].reshape(-1)
        for idx in tqdm(idxs):
            img = imgs[idx-1]
            path = os.path.join(IMAGE_DIR, s, '%05d.png' % (idx))
            io.imsave(path, img)


def extract_labels(labels, splits, SEG40_DIR, SEG13_DIR, save_colored=True):
    mapping40 = loadmat('classMapping40.mat')['mapClass'][0]
    mapping13 = loadmat('class13Mapping.mat')['classMapping13'][0][0][0][0]
    mapping40 = np.insert(mapping40, 0, 0)
    mapping13 = np.insert(mapping13, 0, 0)
    labels = labels.transpose([0, 2, 1])
    labels_40 = mapping40[labels]
    labels_13 = mapping13[labels_40].astype('uint8')

    labels_40 = labels_40.astype('uint8') - 1
    labels_13 = labels_13.astype('uint8') - 1
    #print( np.unique( labels_13 ) )

    if save_colored:
        cmap = colormap()
        os.makedirs('colored_40', exist_ok=True)
        os.makedirs('colored_13', exist_ok=True)

    print("Extracting labels (40 classes)...")

    for s in ['train', 'test']:
        os.makedirs(os.path.join(SEG40_DIR, s), exist_ok=True)
        idxs = splits[s+'Ndxs'].reshape(-1)

        for idx in tqdm(idxs):
            lbl = labels_40[idx-1]
            path = os.path.join(SEG40_DIR, s, '%05d.png' % (idx))
            io.imsave(path, lbl, check_contrast=False)
            if save_colored:
                colored_lbl = cmap[lbl+1]
                io.imsave('colored_40/%05d.png' % idx, colored_lbl)

    print("Extracting labels (13 classes)...")
    for s in ['train', 'test']:
        os.makedirs(os.path.join(SEG13_DIR, s), exist_ok=True)
        idxs = splits[s+'Ndxs'].reshape(-1)

        for idx in tqdm(idxs):
            lbl = labels_13[idx-1]
            path = os.path.join(SEG13_DIR, s, '%05d.png' % (idx))
            io.imsave(path, lbl, check_contrast=False)
            if save_colored:
                colored_lbl = cmap[lbl+1]
                io.imsave('colored_13/%05d.png' % idx, colored_lbl)


def extract_depths(depths, splits, DEPTH_DIR, save_colored=False):
    depths = depths.transpose(0, 2, 1)
    if save_colored:
        os.makedirs('colored_depth', exist_ok=True)
    print("Extracting depths...")
    depths = (depths*1e3).astype(np.uint16)

    for s in ['train', 'test']:
        os.makedirs(os.path.join(DEPTH_DIR, s), exist_ok=True)
        idxs = splits[s+'Ndxs'].reshape(-1)
        for idx in tqdm(idxs):
            depth = depths[idx-1]
            path = os.path.join(DEPTH_DIR, s, '%05d.png' % (idx))
            io.imsave(path, depth, check_contrast=False)

            if save_colored:
                norm = plt.Normalize()
                colored = plt.cm.jet(norm(depth))
                plt.imsave('colored_depth/%05d.png' % (idx), colored)


def extract_normals(normal_zip: str, data_root: str, splits: dict) -> None:
    # prepare directories
    normal_test = Path(data_root) / "normal/test"
    normal_test.mkdir(exist_ok=True, parents=True)
    normal_train = Path(data_root) / "normal/train"
    normal_train.mkdir(exist_ok=True, parents=True)

    # extract raw normals in the form of array
    with TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(normal_zip, 'r') as normal_zip:
            normal_zip.extractall(path=tmpdir)
        with gzip.open(
            f"{tmpdir}/surfacenormal_metadata/all_normals.pklz", "rb"
        ) as f:
            data = pickle.load(f)
    
    # obtain indices of images in each of splits
    reference_test = splits["testNdxs"].reshape(-1)
    reference_train = splits["trainNdxs"].reshape(-1)

    # save raw data in proper directory
    for idx, f_name in enumerate(data["all_filenames"]):
        if int(f_name) in reference_test:
            out_path = normal_test / f"{f_name[1:]}.npy"
        elif int(f_name) in reference_train:
            out_path = normal_train / f"{f_name[1:]}.npy"
        else:
            raise ValueError(f"{f_name} not found in train and test splits!")
        normal_map = data["all_normals"][idx, :]
        np.save(out_path, normal_map)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RYU DATA Extraction')
    parser.add_argument('--mat', type=str, required=True,
                        help='downloaded NYUv2 mat files. http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat')
    parser.add_argument('--data_root', type=str,
                        required=True, help='the output dir')
    parser.add_argument('--save_colored', action='store_true', default=False,
                        help="save colored labels and depth maps for visualization")
    parser.add_argument('--normal_zip', type=str, default=None,
                        help='path to nyuv2_surfacenormal_metadata.zip. https://dl.fbaipublicfiles.com/fair_self_supervision_benchmark/nyuv2_surfacenormal_metadata.zip')

    args = parser.parse_args()

    MAT_FILE = os.path.expanduser(args.mat)
    DATA_ROOT = os.path.expanduser(args.data_root)
    assert os.path.exists(MAT_FILE), "file does not exists: %s" % MAT_FILE

    os.makedirs(DATA_ROOT, exist_ok=True)
    IMAGE_DIR = os.path.join(DATA_ROOT, 'image')
    SEG40_DIR = os.path.join(DATA_ROOT, 'seg40')
    SEG13_DIR = os.path.join(DATA_ROOT, 'seg13')
    DEPTH_DIR = os.path.join(DATA_ROOT, 'depth')
    splits = loadmat('splits.mat')

    os.makedirs(IMAGE_DIR, exist_ok=True)
    os.makedirs(SEG40_DIR, exist_ok=True)
    os.makedirs(SEG13_DIR, exist_ok=True)
    os.makedirs(DEPTH_DIR, exist_ok=True)
    import time
    with h5py.File(MAT_FILE, 'r') as fr:
        images = fr["images"]
        labels = fr["labels"]
        depths = fr["depths"]

        extract_labels(np.array(labels), splits, SEG40_DIR, SEG13_DIR, save_colored=args.save_colored )
        extract_depths(np.array(depths), splits, DEPTH_DIR, save_colored=args.save_colored)
        extract_images(np.array(images), splits, IMAGE_DIR)

        if args.normal_zip is not None and os.path.exists(args.normal_zip):
            extract_normals(args.normal_zip, DATA_ROOT, splits)
        
        if not os.path.exists(os.path.join( DATA_ROOT, 'splits.mat' )):
            shutil.copy2( 'splits.mat', os.path.join( DATA_ROOT, 'splits.mat' ))
