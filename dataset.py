import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import shutil
import random
from PIL import Image
from PIL import ImageFilter
from tqdm import tqdm
import numpy as np
import multiprocessing
import glob

import joblib

class ImageDataset(torch.utils.data.Dataset):
    """Some Information about ImageDataset"""
    def __init__(self, source_dir_pathes=[], cache_dir="./dataset_cache/", size=8, max_len=100000):
        super(ImageDataset, self).__init__()
        self.image_path_list = []
        for dir_path in source_dir_pathes:
            self.image_path_list += glob.glob(os.path.join(dir_path, "*.jpg")) + glob.glob(os.path.join(dir_path, "*.png"))
        self.cache_dir = cache_dir
        self.image_path_list = self.image_path_list[:max_len]
        self.size = -1
        self.max_len = max_len
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        self.set_size(size)
    
    def set_size(self, size):
        if self.size == size:
            return
        self.size = size
        
        print("Initializing cache...")
        # initialize directory
        shutil.rmtree(self.cache_dir)
        # resize image and save to cache directory
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        print("Resizing images... to size: {}".format(size))
        def fn(i):
            img_path = self.image_path_list[i]
            img = Image.open(img_path)
            # get height and width
            H, W = img.size
            if H > W:
                W = int(W * size / H)
                H = size
            else:
                H = int(H * size / W)
                W = size
            img = img.resize((H, W), Image.NEAREST)
            # padding to square
            empty = Image.new("RGB", (size, size), (0, 0, 0))
            # paste image to empty
            empty.paste(img, ((size - H) // 2, (size - W) // 2))
            # save to cache directory
            path = os.path.join(self.cache_dir, str(i) + ".jpg")
            empty.save(path)
            del img
            del empty

        _ = joblib.Parallel(n_jobs=-1)(joblib.delayed(fn)(i) for i in tqdm(range(len(self.image_path_list))))
        print("Resize complete!")

    def __getitem__(self, index):
        # load image
        try:
            img_path = os.path.join(self.cache_dir, str(index) + ".jpg")
            img = Image.open(img_path)
        except Exception:
            print("Skipped error")
            img_path = os.path.join(self.cache_dir,"0.jpg")
            img = Image.open(img_path)
        # to numpy
        img = np.array(img)
        # normalize
        img = img / 127.5 - 1.0
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
            
        return img

    def __len__(self):
        return os.listdir(self.cache_dir).__len__()
