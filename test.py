import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as pyplot
from tqdm import tqdm

import pdb

from dataloader import Salt_dataset, transform
from net import Unet
from utils import rle_encode
    
if __name__ == '__main__':
    dataset = Salt_dataset('./data/test', 'images', None,
         False, 0, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset,
         batch_size=1, shuffle=False, num_workers=8)
         
    print('model initalize')
    net = Unet().cuda()
    net = nn.DataParallel(net, [0, 1])
    print('model load')
    net.load_state_dict(torch.load('./ckpt/unet.pth'))
    net.eval()

    with open('output.csv', 'w') as f:
        f.write('id,rle_mask\n')
        for batch_image, batch_name in tqdm(dataloader):
            outputs = net(batch_image)
            outputs = F.softmax(outputs, dim=1)[:,1,:,:]
            outputs = outputs > 0.95
            # pdb.set_trace()
            for k, v in zip(batch_name, outputs):
                run  = rle_encode(np.array(v))
                f.write('{},{}\n'.format(k[:-4], run))

            pass
