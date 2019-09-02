import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1e-6):
    iflat = pred.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return 1 - ((2. * intersection + smooth) /
                (iflat.sum() + tflat.sum() + smooth))


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = F.sigmoid(pred)
        dice = dice_loss(pred, target)

        loss = bce + dice * self.alpha
        return loss

