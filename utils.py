import os
import torch
import numpy as np
from PIL import Image
import numpy as np

import pdb

def rle_encode(im):
    im = im.transpose()
    pixels = im.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)
