class SlowStartDeeplabV3Scheduler(object):
    r"""Slow start learning rate scheduler.

    ```
    current_slow_start_lr = slow_start_lr + (step/slow_start_step) * (base_lr - slow_start_lr)
    ```
    
    Reference: 
        1. https://github.com/tensorflow/models/blob/28f6182fc9afaf11104a5abe7c21b57b6aeb30e2/research/deeplab/utils/train_utils.py#L268
        2. https://github.com/Jiaming-Liu/pytorch-lr-scheduler"""
    def __init__(self, optimizer, base_lr, 
                 slow_start_lr, 
                 slow_start_step):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.slow_start_lr = slow_start_lr
        self.slow_start_step = slow_start_step
        self.current_step = 0

    def step(self):
        self.current_step += 1
        adjusted_slow_start_lr = self.__adjust_slow_start_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = adjusted_slow_start_lr

    def __adjust_slow_start_lr(self):
        adjusted_slow_start_lr = (self.slow_start_lr
                               + (self.base_lr - self.slow_start_lr)
                               * self.current_step / self.slow_start_step)
        return adjusted_slow_start_lr