import os
import torch
import numpy as np
from multiprocessing import Pool
from itertools import repeat
from functools import partial
from PIL import Image

class Data_loader:
    def __init__(self):
        self.train_list = []
        self.val_list = []
        self.train_size = []
        self.val_size = []
        
    def make_file_list(self, dir_path, ext_orig, ext_mask, val_rate=0.8, shuffle=True):
        file_list = []
        for path, _, files in os.walk(os.path.join(dir_path, 'images')):
            for file_ in files:
                name, ext = os.path.splitext(file_)
                file_list.append((name + ext_orig, name + ext_mask))
        cutline = int(len(file_list) * val_rate)
        self.train_list, self.val_list = file_list[:-cutline], file_list[-cutline:]
        self.train_size, self.val_size = len(self.train_list), len(self.val_list)
        return self

    def make_batch_from_file_list(self, path_prefix, path_orig, path_mask, is_train=True, batch_size=1,  num_processes=1):
        p = Pool(num_processes)

        if is_train:
            file_list = self.train_list
        else:
            file_list = self.train_list

        for i in range(0, len(file_list), batch_size):
            batch_orig = p.map(partial(self.make_batch_orig, path_prefix=path_prefix, path_orig=path_orig), file_list[i:i+batch_size])
            batch_mask = p.map(partial(self.make_batch_mask, path_prefix=path_prefix, path_mask=path_mask), file_list[i:i+batch_size])
            yield np.array(batch_orig), np.array(batch_mask)
            
    def make_batch_orig(self, file_, path_prefix, path_orig):
        orig = np.array(Image.open(os.path.join(path_prefix, path_orig, file_[0])).resize((224, 224), resample=Image.NEAREST))
        return orig

    def make_batch_mask(self, file_, path_prefix, path_mask):
        mask = np.array(Image.open(os.path.join(path_prefix, path_mask, file_[1])).resize((224, 224), resample=Image.NEAREST))
        return mask