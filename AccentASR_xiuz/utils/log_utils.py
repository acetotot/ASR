from tensorboardX import SummaryWriter
import numpy as np


class AveMeter:
    def __init__(self):
        self.items = []

    def add_item(self, i):
        self.items.append(i)

    def get_averaged_result(self):
        a = np.asarray(self.items)
        return np.average(a)


class Logger:
    def __init__(self, logdir):
        self.log = SummaryWriter(logdir)

    def write_log(self, val_name, val_dict, step):
        '''Write log to TensorBoard'''
        if 'img' in val_name:
            self.log.add_image(val_name, val_dict, step)
        elif 'txt' in val_name or 'hyp' in val_name:
            self.log.add_text(val_name, val_dict, step)
        else:
            self.log.add_scalar(val_name, val_dict, step)
