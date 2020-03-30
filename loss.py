import sys
from torch import nn
import torch


class DiceLoss(nn.Module):
    """
    Dice loss function class
    """
    def __init__(self, squared_denom=False):
        super(DiceLoss, self).__init__()
        self.smooth = sys.float_info.epsilon
        self.squared_denom = squared_denom

    def forward(self, x, target):
        x = x.view(-1)
        target = target.view(-1)
        intersection = (x * target).sum()
        numer = 2. * intersection + self.smooth
        factor = 2 if self.squared_denom else 1
        denom = x.pow(factor).sum() + target.pow(factor).sum() + self.smooth
        dice_index = numer / denom
        return 1 - dice_index
