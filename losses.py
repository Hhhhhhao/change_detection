import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_binary_cross_entropy(output, target, weight=None):
    output = torch.sigmoid(output)
    if weight is not None:

        loss = weight * (target * torch.log(output)) + \
               (1 - weight) * ((1 - target) * torch.log(1 - output))
    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


def dice_loss(pred, target, smooth=1e-2):
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target, weight):
        bce = weighted_binary_cross_entropy(pred, target, weight)

        pred = F.sigmoid(pred)
        dice = dice_loss(pred, target)

        loss = bce * self.alpha + dice * (1 - self.alpha)
        return loss