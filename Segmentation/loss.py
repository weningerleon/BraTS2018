import torch
from torch.nn.modules.loss import _Loss

class DiceLoss(_Loss):
    def __init__(self, size_average=True, reduce=True):
        super(DiceLoss, self).__init__(size_average, reduce)

    def forward(self, pred, target):
        return 1 - dice_score(pred, target)


def dice_score(pred, target):

    epsilon = 0.00001

    num_labels = pred.shape[1]
    dice = torch.tensor([0.]).cuda()
    for i in range(num_labels):
        pred_i = pred[:,i,:,:,:]
        target_i = target[:,i,:,:,:]

        tp = (pred_i * target_i).sum()
        t = (pred_i**2).sum() + (target_i**2).sum()

        dice_i = (2*tp + epsilon) / (t + epsilon)

        dice = dice + dice_i / num_labels

    return dice

