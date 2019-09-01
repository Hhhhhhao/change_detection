import numpy as np
import torch
from torch.utils.data import DataLoader
from functools import partial
from datasets import RssraiDataset


def collate_fn(data, batch_size, img_size):
    img, mask = data[0]
    batch_x, batch_y = [], []

    # # randomly crop image and mask at white pixels
    # pixels = get_pixels(mask[0], img_size, img_size)
    # num_pix = len(pixels[0])

    # while num_img < batch_size:
        # if num_pix == 0:
        #     s_x = np.random.randint(0, img.shape[1] - img_size + 1)
        #     s_y = np.random.randint(0, img.shape[2] - img_size + 1)
        # else:
        #     index = np.random.randint(num_pix)
        #     s_x = pixels[0][index]
        #     s_y = pixels[1][index]

    for i in range(batch_size):
        s_x = np.random.randint(0, img.shape[1] - img_size + 1)
        s_y = np.random.randint(0, img.shape[2] - img_size + 1)
        y = mask[:, s_x:s_x + img_size, s_y:s_y + img_size]
        x = img[:, s_x:s_x + img_size, s_y:s_y + img_size]
        batch_x.append(x)
        batch_y.append(y)

    batch_x = np.array(batch_x)
    batch_y = np.array(batch_y)

    if torch.cuda.is_available():
        batch_x = torch.cuda.FloatTensor(batch_x)
        batch_y = torch.cuda.FloatTensor(batch_y)
    else:
        batch_x = torch.FloatTensor(batch_x)
        batch_y = torch.FloatTensor(batch_y)

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


if __name__ == '__main__':
    data_loader = RssraiDataLoader(which_set='train', batch_size=16, img_size=256, shuffle=True)

    for i, (input, mask) in enumerate(data_loader):
        print('{}th batch: input shape {}, mask shape {}'.format(i, input.shape, mask.shape))

