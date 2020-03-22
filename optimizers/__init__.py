import torch
from optimizers.slow_start_lr_deeplabv3_scheduler import SlowStartDeeplabV3Scheduler


def get_segmentation_optimizer(model, max_iter, base_lr=0.001, power=0.9):
    r'''generate SGD optimizer and scheduler'''
    # used by: Deeplab, PSPNet, SegNet
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, 
                                momentum=0.9, weight_decay=0.0)
    lr_update = lambda iter: (1 - iter/max_iter)**power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, scheduler

def get_captioning_optimizer(model, base_lr=5e-4):
    r'''generate Adam optimizer and StepLR scheduler'''
    # used by: Neural Baby Talk, Grounded Video Captioning
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.8)
    return optimizer, scheduler

def get_quick_optimizer(model, max_iter, base_lr=0.001, power=0.9):
    r'''generate Adam optimizer and StepLR scheduler'''
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    lr_update = lambda iter: (1 - iter/max_iter)**power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, scheduler

def get_quick_optimizer_multi(models, max_iter, base_lr=0.001, power=0.9):
    r'''generate Adam optimizer and StepLR scheduler'''
    optimizer = torch.optim.Adam(sum([list(m.parameters()) for m in models], []), lr=base_lr)
    lr_update = lambda iter: (1 - iter/max_iter)**power
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_update)
    return optimizer, scheduler

def get_arcface_casia_optimizer(model, base_lr=0.1):
    r'''generate SGD optimizer and StepLR scheduler for ArcFace. Max iter is 32K
    this is also used in SphereFace'''
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, 
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20000, 28000], gamma=0.1)
    return optimizer, scheduler

def get_arcface_ms1mv2_optimizer(model, base_lr=0.1):
    r'''generate SGD optimizer and StepLR scheduler for ArcFace. Max iter is 180K'''
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, 
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100000, 160000], gamma=0.1)
    return optimizer, scheduler

def get_cosface_small_dataset_optimizer(model, base_lr=0.1):
    r'''generate SGD optimizer and StepLR scheduler for CosFace. Max iter is 30K'''
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, 
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[16000, 24000, 28000], gamma=0.1)
    return optimizer, scheduler

def get_cosface_large_dataset_optimizer(model, base_lr=0.05):
    r'''generate SGD optimizer and StepLR scheduler for CosFace. Max iter is 240K'''
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, 
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80000, 140000, 200000], gamma=0.1)
    return optimizer, scheduler

def get_lsoftmax_cifar_optimizer(model, base_lr=0.1):
    r'''generate SGD optimizer and StepLR scheduler for L-Softmax. Max iter is 18K'''
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, 
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[12000, 15000], gamma=0.1)
    return optimizer, scheduler

def get_lsoftmax_face_optimizer(model, base_lr=0.1):
    r'''generate SGD optimizer and StepLR scheduler for L-Softmax. Max iter is 30 epochs'''
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr, 
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)
    return optimizer, scheduler