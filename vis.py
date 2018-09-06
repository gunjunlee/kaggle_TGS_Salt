import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from math import ceil
import pdb
import os
import time
import random

from utils import *
from dataloader import Salt_dataset, transform
from net import Unet
from metric import dice_loss, iou
from termcolor import colored

VAL_RATIO = 0.1
BATCH_SIZE = 1
NUM_PROCESSES = 8

if __name__ == '__main__':
    dataset = {phase: Salt_dataset('./data/train', 'images', 'masks',
         phase=='train', VAL_RATIO, transform) for phase in ['train', 'val']}
    dataloader = {phase: torch.utils.data.DataLoader(dataset[phase],
         batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
         for phase in ['train', 'val']}

    net = Unet().cuda()
    net = nn.DataParallel(net, [0, 1])
    net.load_state_dict(torch.load('./ckpt/unet.pth'))

    # val
    net.eval()
    for batch_image, batch_mask in tqdm(dataloader['val']):
        batch_image = batch_image.cuda()
        batch_mask = batch_mask.cuda()
        outputs = net(batch_image)
        outputs = F.softmax(outputs, dim=1)[:,1,:,:]
        outputs = outputs > 0.95

        plt.imshow(outputs.cpu().numpy()[0])
        plt.show()
        plt.imshow(batch_mask.cpu().numpy()[0])
        plt.show()
