import torch
import torch.nn as nn


def hard_pixel_loss_sampling(loss, k=100):
    r"""Hard pixel loss-sampling.
    
    Reference: https://github.com/tensorflow/models/blob/28f6182fc9afaf11104a5abe7c21b57b6aeb30e2/research/deeplab/utils/train_utils.py#L33
    
    Args:
        loss (torch.tensor): The loss tensor, from which losses are sampled. This is a tensor of shape (n_samples, n_classes, ...)
        k (int): Number of loss samples to return.
    Returns:
        torch.tensor: Tensor of shape (n_samples, n_classes, k). This is the sampled pixels.
    """
    loss = loss.reshape((loss.shape[0], loss.shape[1], -1))
    top_k_pixels, _ = torch.topk(loss, k)
    return top_k_pixels

def MIT_group_weight(module):
    r"""MIT CSAIL group weighting, i.e. weight decay and not weight decay. In this setting, only weight matrices of nn.Linear or nn.conv._ConvNd is applied weight decay.

    Reference: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/train.py

    Args:
        module (torch.nn.Module): The module whose parameters are to be divided into groups for training.
    Returns:
        list: List of Python dict. Each element specifies a parameter group and its corersponding training configuration, i.e. parameters for torch.optim.Optimizer.
    """
    group_decay, group_no_decay = [], []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [
        dict(params=group_decay), 
        dict(params=group_no_decay, weight_decay=0.0)
    ]
    return groups

def MIT_init_weights(m):
    r"""MIT CSAIL weight initialization. This is to be used in `models.apply(MIT_init_weights)`. In this setting, Conv layers are initialized with Kaiming normal strategy, and BN layers are initialize with unit-weight and zero-bias.
    
    Reference: https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/models/models.py
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)

def UNET_init_weights(m):
    r"""UET AIL weight initialization. This is used in `models.apply(UNET_init_weights`. In this setting, we init weights of a layer with Xavier initialization"""
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias") is not None:
            m.bias.data.fill_(0.01)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias") is not None:
            m.bias.data.fill_(0.01)

def get_lr(optimizer):
    r"""Get learning rates of parameter groups of a torch.optim.Optimizer"""
    lr = [param_group['lr'] for param_group in optimizer.param_groups]
    return lr