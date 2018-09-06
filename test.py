import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as pyplot

import pdb

from dataloader import Salt_dataset, transform
from net import Unet

BATCH_SIZE = 1
    
if __name__ == '__main__':
    dataset = Salt_dataset('./data/test', 'images', None,
         False, 0, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset,
         batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
         
    net = Unet().cuda()
    net = nn.DataParallel(net, [0, 1])
    net.load_state_dict(torch.load('./ckpt/unet.pth'))
    net.eval()

    for batch_image, _ in dataloader:
        outputs = net(batch_image)
        outputs = F.softmax(outputs, dim=1)[:,1,:,:]
        outputs = outputs > 0.95
        pdb.set_trace()
        pass