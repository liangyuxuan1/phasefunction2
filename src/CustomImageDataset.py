import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io as scipyIO
import pandas as pd
import os

# PyTorch offers domain-specific libraries such as TorchText, TorchVision, and TorchAudio, all of which include datasets. 
# For this tutorial, we will be using a TorchVision dataset.
# The torchvision.datasets module contains Dataset objects for many real-world vision data like CIFAR, COCO (full list here). 

# Creating a Custom Dataset for your files
# A custom Dataset class must implement three functions: __init__, __len__, and __getitem__. 

# data = io.loadmat(fullFileName, variable_names='rawData', mat_dtype=True)

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0]) # 0: filename
        # image = read_image(img_path)
        image = scipyIO.loadmat(img_path).get('rawData')
        image = image.astype(np.float64)
        h, w = image.shape
        image = torch.from_numpy(image).reshape(1, h, w)
        image = image.float()

        ua = self.img_labels.iloc[idx, 1]    # 1: ua value
        us = self.img_labels.iloc[idx, 2]    # 2: us value
        g = self.img_labels.iloc[idx, 3]     # 3: g value

        gt = torch.tensor([ua, us, g])
        # gt = torch.tensor(gt).reshape(-1)
        gt = gt.float()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt = self.target_transform(gt)

        return image, gt