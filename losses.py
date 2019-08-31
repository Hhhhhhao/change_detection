import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(BCEDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)

        pred = F.sigmoid(pred)
        dice = dice_loss(pred, target)

        loss = bce * self.alpha + dice * (1 - self.alpha)
        return loss