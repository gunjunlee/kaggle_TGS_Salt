import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as pyplot
from tqdm import tqdm
from PIL import Image

import pdb

from dataloader import Salt_dataset
from models.unet import Unet
from models.linknet import LinkNet34
from utils import rle_encode
    
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

if __name__ == '__main__':
    dataset = Salt_dataset('./data/test', 'images', None,
         False, 0, transform=transform)
    
    dataloader = torch.utils.data.DataLoader(dataset,
         batch_size=1, shuffle=False, num_workers=8)
         
    print('model initalize')
    net = LinkNet34(num_channels=3, num_classes=1, pretrained=True)
    net = nn.DataParallel(net.cuda())
    print('model load')
    net.load_state_dict(torch.load('./ckpt/unet-bn-inch1-plateau.pth'))
    net.eval()
    # pdb.set_trace()
    with open('output/result.csv', 'w') as f:
        f.write('id,rle_mask\n')
        for batch_image, batch_name in tqdm(dataloader):
            outputs = net(batch_image).squeeze(dim=1)
            outputs = torch.sigmoid(outputs)
            # pdb.set_trace()
            outputs = outputs > 0.30
            # pdb.set_trace()
            for k, v in zip(batch_name, outputs):
                # pdb.set_trace()
                v = transforms.functional.to_pil_image(v.unsqueeze(dim=0).cpu())\
                    .resize((101,101), resample=Image.NEAREST)
                run  = rle_encode(np.array(v).transpose())
                f.write('{},{}\n'.format(k[:-4], run))

            pass
