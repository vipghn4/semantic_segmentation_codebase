import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def CELoss(y_pred, onehot_masks, eps=1e-9):
    r"""Compute Cross-Entropy loss.

    Args:
        y_pred (torch.tensor): Model prediction, which is a tensor of shape (n_samples, n_class, h, w).
        masks (torch.tensor): Groundtruth, which is a tensor of shape (n_samples, n_class, h, w).
    Returns
        torch.tensor: (float) loss value
    """
    y_pred = F.softmax(y_pred, dim=1)
    loss = torch.sum(- onehot_masks * torch.log(y_pred + eps), dim=1).mean()
    return loss

def IoU(y_pred, onehot_masks, threshold=0.5, eps=1e-9):
    r'''Compute mean IoU using prediction and target'''
    y_pred, y_true = y_pred.float(), onehot_masks.float()
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def DICE(y_pred, onehot_masks, threshold=0.5, eps=1e-9):
    r'''Compute mean DICE using prediction and target'''
    y_pred, y_true = y_pred.float(), onehot_masks.float()
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3))
    dice = (2*intersection + eps) / (union + eps)
    return dice.mean()