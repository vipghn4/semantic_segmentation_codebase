import tensorflow as tf
import torch
import torch.nn.functional as F


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