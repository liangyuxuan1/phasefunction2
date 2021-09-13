import torch
from torch.utils.data import Dataset
from preprocessing import DataPreprocessor

class CustomImageDataset_Pickle(Dataset):
    def __init__(self, img_labels, file_preprocessed, transform=None, target_transform=None):
        self.img_labels = img_labels
        self.images = DataPreprocessor().load(file_preprocessed)
        # self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        ua = self.img_labels.iloc[idx, 1]    # 1: ua value
        us = self.img_labels.iloc[idx, 2]    # 2: us value
        g = self.img_labels.iloc[idx, 3]     # 3: g value

        gt = torch.tensor([ua, us, g])
        gt = gt.float()

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            gt = self.target_transform(gt)

        return image, gt