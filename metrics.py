import torch
import torch.nn as nn
import torch.nn.functional as F


def IoU(y_pred, masks, onehot_masks, threshold=0.5):
    r'''Compute mean IoU using prediction and target'''
    y_pred, y_true = y_pred.float(), onehot_masks.float()
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()

def DICE(y_pred, masks, onehot_masks, threshold=0.5):
    r'''Compute mean DICE using prediction and target'''
    y_pred, y_true = y_pred.float(), onehot_masks.float()
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3))
    dice = (2*intersection + 1e-6) / (union + 1e-6)
    return dice.mean()