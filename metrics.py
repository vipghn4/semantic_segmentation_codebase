import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def CELoss(y_pred, onehot_masks, eps=1e-9):
    r"""Compute Cross-Entropy loss.

    Args:
        y_pred (torch.tensor): Model prediction, which is a tensor of shape (n_samples, n_class, h, w).
        onehot_masks (torch.tensor): Groundtruth, which is a tensor of shape (n_samples, n_class, h, w).
    Returns
        torch.tensor: (float) loss value.
    """
    y_pred = F.softmax(y_pred, dim=1)
    loss = torch.sum(- onehot_masks * torch.log(y_pred + eps), dim=1).mean()
    return loss

def accuracy(y_pred, onehot_masks):
    r"""Compute accuracy of segmentation result.

    Args:
        y_pred (torch.tensor): Model prediction, which is a tensor of shape (n_samples, n_class, h, w).
        onehot_masks (torch.tensor): Groundtruth, which is a tensor of shape (n_samples, n_class, h, w).
    Returns
        torch.tensor: (float) accuracy value.
    """
    preds = torch.argmax(y_pred, axis=1)
    label = torch.argmax(onehot_masks, axis=1)
    acc_sum = torch.sum((preds == label).long())
    valid_sum = label.numel()
    acc = acc_sum.float() / (valid_sum + 1e-10)
    return acc, valid_sum

def pixel_acc(y_pred, onehot_masks):
    r"""Compute pixel accuracy of segmentation result.

    Args:
        y_pred (torch.tensor): Model prediction, which is a tensor of shape (n_samples, n_class, h, w).
        onehot_masks (torch.tensor): Groundtruth, which is a tensor of shape (n_samples, n_class, h, w).
    Returns
        torch.tensor: (float) pixel accuracy value.
    """
    preds = torch.argmax(y_pred, axis=1)
    label = torch.argmax(onehot_masks, axis=1)
    acc_sum = torch.sum((preds == label).long())
    pixel_sum = label.numel()
    acc = acc_sum.float() / (pixel_sum + 1e-10)
    return acc

def intersectionAndUnion(y_pred, onehot_masks):
    r"""Compute intersection and union of prediction and target.

    Args:
        y_pred (torch.tensor): Model prediction, which is a tensor of shape (n_samples, n_class, h, w).
        onehot_masks (torch.tensor): Groundtruth, which is a tensor of shape (n_samples, n_class, h, w).
    Returns
        torch.tensor: FloatTensor of shape (n_class, ), which is the intersection values.
        torch.tensor: FloatTensor of shape (n_class, ), which is the union values.
    """
    y_pred, y_true = y_pred.float(), onehot_masks.float()
    intersection = (y_true * y_pred).sum((0, 2, 3))
    union = y_true.sum((0, 2, 3)) + y_pred.sum((0, 2, 3)) - intersection
    return intersection, union

def IoU(y_pred, onehot_masks, threshold=0.5, eps=1e-9):
    r"""Compute mean IoU using prediction and target.

    Args:
        y_pred (torch.tensor): Model prediction, which is a tensor of shape (n_samples, n_class, h, w).
        onehot_masks (torch.tensor): Groundtruth, which is a tensor of shape (n_samples, n_class, h, w).
    Returns
        torch.tensor: (float) IoU value.
    """
    y_pred, y_true = y_pred.float(), onehot_masks.float()
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean()

def DICE(y_pred, onehot_masks, threshold=0.5, eps=1e-9):
    r"""Compute mean DICE using prediction and target.
    
    Args:
        y_pred (torch.tensor): Model prediction, which is a tensor of shape (n_samples, n_class, h, w).
        onehot_masks (torch.tensor): Groundtruth, which is a tensor of shape (n_samples, n_class, h, w).
    Returns
        torch.tensor: (float) DICE value.
    """
    y_pred, y_true = y_pred.float(), onehot_masks.float()
    intersection = (y_true * y_pred).sum((2, 3))
    union = y_true.sum((2, 3)) + y_pred.sum((2, 3))
    dice = (2*intersection + eps) / (union + eps)
    return dice.mean()