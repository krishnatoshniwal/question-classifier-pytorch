import pickle as pkl


def save_obj(file_name, obj):
    file_name = file_name if '.pkl' in file_name else file_name + '.pkl'
    with open(file_name, 'wb') as file:
        pkl.dump(obj, file)


def load_obj(file_name):
    file_name = file_name if '.pkl' in file_name else file_name + '.pkl'
    with open(file_name, 'rb') as file:
        obj = pkl.load(file)
    return obj


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self._reset()

    def _reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clean_line():
    # pass
    print(' ' * 80, end='\r')


def clean_print(*args, **kwargs):
    clean_line()
    print(*args, **kwargs)