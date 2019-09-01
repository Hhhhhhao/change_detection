import torch
import torch.nn as nn
import torch.nn.functional as F


def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()


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
        pred = F.sigmoid(pred)

        bce = weighted_binary_cross_entropy(pred, target, weight)
        dice = dice_loss(pred, target)

        loss = bce + dice * self.alpha
        return loss
    