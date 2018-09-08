import torch
import torch.nn.functional as F
import numpy as np
import pdb

def dice_loss(pred, target):

    pred = pred.float()
    target = target.float()
    smooth = 1e-4

    p = F.softmax(pred, dim=1)[:,1,:,:]
    
    inter = (target*p).sum(dim=2).sum(dim=1)
    dim1 = (p).sum(dim=2).sum(dim=1)
    dim2 = (target).sum(dim=2).sum(dim=1)

    coeff = (2 * inter + smooth) / (dim1 + dim2 + smooth)
    dice_total = 1-coeff.sum(dim=0)/coeff.size(0)
    # import pdb
    # pdb.set_trace()
    return dice_total

def iou(pred, target):
    pred = pred.float()
    target = target.float()
    smooth = 1e-4

    candidate = []
    for threshold in np.arange(0.5, 1, 0.05):
        p = F.softmax(pred, dim=1)[:,1,:,:] > threshold
        p = p.float()

        inter = (target*p).sum(dim=2).sum(dim=1)
        dim1 = (p).sum(dim=2).sum(dim=1)
        dim2 = (target).sum(dim=2).sum(dim=1)

        coeff = (inter + smooth) / (dim1 + dim2 - inter + smooth)
        iou_ = coeff.sum(dim=0)/coeff.size(0)
        candidate.append(iou_)
        
    return np.array(candidate)

if __name__ == '__main__':
    a = torch.from_numpy(
            np.array([
                    [
                        [[1, 1], [1, -1e+8]],
                        [[-1e+8, -1e+8], [-1e+8, 1]]
                    ]
                ])
        )
    b = torch.from_numpy(np.array([[[1, 1], [1, 1]]]))

    print(dice_loss(a, b))
    print(iou(a, b))
    pass