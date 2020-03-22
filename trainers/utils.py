import tensorflow as tf
import torch
import torch.nn.functional as F


class AverageMeter(object):
    r"""Average Meter implemented by MIT CSAIL. Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def update(self, val, weight=1):
        r"""Append a new value to the AverageMeter
        
        Args:
            val (object): Initial value.
            weight (float): Weight assigned to the initial value to carry out average and summation.
        """
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def __initialize(self, val, weight):
        r"""Initialize Average meter by a value"""
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def __add(self, val, weight):
        r"""Append a value to AverageMeter"""
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        r"""Get current value"""
        return self.val

    def average(self):
        r"""Get average value"""
        return self.avg


class Logger(object):
    def __init__(self, log_dir):
        r"""Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        r"""Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)
            self.writer.flush()

    def list_of_scalars_summary(self, tag_value_pairs, step):
        r"""Log scalar variables."""
        with self.writer.as_default():
            for tag, value in tag_value_pairs:
                tf.summary.scalar(tag, value, step=step)
            self.writer.flush()


class bcolors:
    '''ANSI terminal output formatting'''
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def log(x):
        return f"{bcolors.BOLD}{bcolors.OKBLUE}{x}{bcolors.ENDC}{bcolors.ENDC}"


def init_weights(m):
    r'''init weights of a layer with xavier initialization'''
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias") is not None:
            m.bias.data.fill_(0.01)
    elif type(m) == torch.nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if getattr(m, "bias") is not None:
            m.bias.data.fill_(0.01)

def logits_to_onehot(logits):
    probs = F.softmax(logits, dim=1)
    max_idx = torch.argmax(probs, dim=1, keepdim=True)
    onehot_pred_masks = torch.FloatTensor(probs.shape).to(probs.device)
    onehot_pred_masks.zero_()
    onehot_pred_masks.scatter_(1, max_idx, 1)
    return onehot_pred_masks