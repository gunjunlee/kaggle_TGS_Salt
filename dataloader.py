import os
import torch
import numpy as np
import torch
import torch.utils.data
from PIL import Image
from math import ceil

class Data_loader(torch.utils.data.Dataset):
    def __init__(self, dir_root, dir_image, dir_mask, is_train, val_rate, transform=None):
        self.dir_root = dir_root
        self.dir_image = dir_image
        self.dir_mask = dir_mask
        self.is_train = is_train
        self.val_rate = val_rate
        self.transform = transform
        self.file_list = []
        
        for path, _, files in os.walk(os.path.join(dir_root, dir_image)):
            for file_ in files:
                self.file_list.append(file_)

        cut = ceil(len(self) * val_rate)
        if self.is_train:
            self.file_list = self.file_list[:-cut]
        else:
            self.file_list = self.file_list[-cut:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.dir_root, self.dir_image, self.file_list[idx]))
        mask = Image.open(os.path.join(self.dir_root, self.dir_mask, self.file_list[idx]))
        
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)
        
        return sample


def transform(sample):
    image = sample['image']
    mask = sample['mask']
    
    return {'image': image, 'mask': mask}