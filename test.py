import os
import argparse
from glob import glob
import warnings

import numpy as np
from tqdm import tqdm

from skimage.io import imread, imsave

import torch

import models
from metrics import dice_coef, batch_iou, mean_iou, iou_score
import losses
from utils import count_params
from data_loaders import RssraiDataLoader


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl' %val_args.name)

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = models.__dict__[args.arch](args.in_ch, args.out_ch, args.num_filters)

    if torch.cuda.is_available():
        model = model.cuda()

    model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model.eval()

    val_loader = RssraiDataLoader(
        which_set='test',
        batch_size=args.batch_size,
        img_size=args.img_size,
        shuffle=False
    )

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        with torch.no_grad():
            for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                # compute output
                output = model(input)[-1]
                output = torch.sigmoid(output).data.cpu().numpy()
                img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]

                for i in range(output.shape[0]):
                    imsave('output/%s/'%args.name+os.path.basename(img_paths[i]), (output[i,0,:,:]*255).astype('uint8'))

        torch.cuda.empty_cache()

    # IoU
    ious = []
    for i in tqdm(range(len(val_mask_paths))):
        mask = imread(val_mask_paths[i])
        pb = imread('output/%s/'%args.name+os.path.basename(val_mask_paths[i]))

        mask = mask.astype('float32') / 255
        pb = pb.astype('float32') / 255


        '''
        plt.figure()
        plt.subplot(121)
        plt.imshow(mask)
        plt.subplot(122)
        plt.imshow(pb)
        plt.show()
        '''

        iou = iou_score(pb, mask)
        ious.append(iou)
    print('IoU: %.4f' %np.mean(ious))


if __name__ == '__main__':
    main()