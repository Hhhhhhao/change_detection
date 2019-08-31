import os
import numpy as np
from tifffile import imread
from torch.utils.data import Dataset
from utils import get_files


class RssraiDataset(Dataset):
    """
    rssrai2019 change detection dataset
    """
    def __init__(self, which_set):
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data/rssrai2019_change_detection', which_set)
        assert os.path.exists(self.data_dir), "cannot find data folder: {}".format(self.data_dir)

        self.img1_ids = get_files(os.path.join(self.data_dir, 'img_2017/'))
        self.img2_ids = get_files(os.path.join(self.data_dir, 'img_2018/'))
        self.mask_ids = get_files(os.path.join(self.data_dir, 'mask/'))

    def __len__(self):
        return len(self.img1_ids)

    def __getitem__(self, idx):
        img1_id = self.img1_ids[idx]
        img2_id = self.img2_ids[idx]
        mask_id = self.mask_ids[idx]
        img1 = imread(img1_id).astype('uint8')
        img2 = imread(img2_id).astype('uint8')
        mask = imread(mask_id).astype('uint8')

        img1 = img1.astype('float32') / 255.
        img2 = img2.astype('float32') / 255.
        mask = mask.astype('float32') / 255.

        img = np.concatenate([img1, img2], axis=-1)
        img = img.transpose((2, 0, 1))
        mask = mask[:, :, np.newaxis]
        mask = mask.transpose((2, 0, 1))
        return img, mask


if __name__ == '__main__':
    from tqdm import tqdm
    import matplotlib.pyplot as plt
    import cv2

    train_dataset = RssraiDataset(which_set='train')
    print('length of the dataset: {}'.format(len(train_dataset)))

    for i, (input, mask) in tqdm(enumerate(train_dataset), total=len(train_dataset)):
        print('input image shape:{}'.format(input.shape))
        print('mask shape:{}'.format(mask.shape))
        input = input.transpose(1, 2, 0)

        plt.imshow((input[:, :, :3] * 255).astype('uint8'))
        plt.show()

        cv2.imshow('image', input[:, :, :3] * 255)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()

        break

