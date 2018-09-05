import os
import torch
import numpy as np
from PIL import Image

def normalize(img_batch, mean, std):
    for t, m, s in zip(img_batch, mean, std):
        t.div_(255).sub_(m).div_(s)
    return img_batch

def to_tensor_img(img_batch):
    """
    convert np.ndarray img batch to torch.floattensor

    Parameters
    ----------
    img_batch : np.nparray
        img batch
    
    """

    tensor = torch.FloatTensor(img_batch).permute((0, 3, 1, 2))
    return tensor

def to_tensor_mask(mask_batch):
    """
    convert np.ndarray mask batch to torch.floattensor

    Parameters
    ----------
    mask_batch : np.nparray
        mask batch
    
    """

    tensor = torch.LongTensor(mask_batch)
    return tensor