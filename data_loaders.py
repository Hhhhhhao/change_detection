import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
from datasets import RssraiDataset


def collate_fn(data, batch_size, img_size):
    img, mask = data[0]
    batch_x, batch_y = [], []

    for i in range(batch_size):
        # randomly crop image and mask
        s_x = np.random.randint(0, img.shape[1] - img_size)
        s_y = np.random.randint(0, img.shape[2] - img_size)
        x = img[:, s_x:s_x + img_size, s_y:s_y + img_size]
        y = mask[:, s_x:s_x + img_size, s_y:s_y + img_size]
        batch_x.append(x)
        batch_y.append(y)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    if torch.cuda.is_available():
        batch_x = torch.cuda.FloatTensor(batch_x)
        batch_y = torch.cuda.LongTensor(batch_y)
    else:
        batch_x = torch.FloatTensor(batch_x)
        batch_y = torch.LongTensor(batch_y)

    return batch_x, batch_y


class RssraiDataLoader(DataLoader):
    """
    Retinal vessel segmentation data loader
    """
    def __init__(self,
                 which_set='train',
                 batch_size=16,
                 img_size=256,
                 shuffle=True
                 ):
        self.dataset = RssraiDataset(which_set=which_set)
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle

        super(RssraiDataLoader, self).__init__(
            dataset=self.dataset,
            batch_size=1, # batch_size set to 1 as we use only 1 full images to extract many patches
            shuffle=self.shuffle,
            num_workers=0,
            drop_last=self.shuffle,
            collate_fn=partial(collate_fn, batch_size=self.batch_size, img_size=self.img_size))

