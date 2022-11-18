import logging
import os
import shutil
import sys

import PIL.Image as Image
import torch.utils.data as data
import torchvision.transforms as transforms


class MemoryFriendlyLoader(data.Dataset):
    def __init__(self, img_dir, task):
        self.img_dir = img_dir
        self.task = task
        self.names = []

        for root, _, names in os.walk(self.img_dir):
            for name in names:
                self.names.append(os.path.join(root, name))

        self.names.sort()
        self.count = len(self.names)

        transform_list = []
        transform_list += [transforms.ToTensor()]
        self.transform = transforms.Compose(transform_list)

    def __getitem__(self, index):
        img = Image.open(self.names[index]).convert('RGB')
        img = self.transform(img)
        img_name = self.names[index].split('/')[-1]
        return img, img_name

    def __len__(self):
        return self.count


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file) 


def setup_logger(name, logfile, formatter, streamhandler=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # fh: file handler
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.INFO)
    fh_formatter = logging.Formatter(formatter)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    if streamhandler:
        # ch: console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter(formatter)
        ch.setFormatter(ch_formatter)
        logger.addHandler(ch)

    return logger
