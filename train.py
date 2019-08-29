import os
import argparse
import pandas as pd
from tqdm import tqdm
from collections import OrderedDict
from datetime import datetime
from sklearn.externals import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import models
import losses
from data_loaders import RssraiDataLoader
from metrics import iou_score
from utils import count_params, save_example

arch_names = list(models.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default='baseline',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--img_size', default=256, help='size of training image patches')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')
    parser.add_argument('--in_ch', default=8, type=int,
                        help='input channels')
    parser.add_argument('--out_ch', default=1, type=int,
                        help='output channels')
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=20, type=int,
                        metavar='N', help='early stopping (default: 20)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    args = parser.parse_args()

    return args


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer):
    losses = AverageMeter()
    ious = AverageMeter()

    model.train()

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for i, (input, target) in pbar:
        # compute output
        outputs = model(input)
        loss = 0
        for output in outputs:
            loss += criterion(output, target)
        loss /= len(outputs)
        iou = iou_score(outputs[-1], target)

        # update log and progress bar
        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        pbar.set_postfix({'loss': loss.item(), 'iou': iou})

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        break

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
    ])

    return log


def main():
    args = parse_args()
    timestamp = datetime.now().strftime('%m%d_%H%M%S')
    model_dir = 'models/{}'.format(args.name + '_' + timestamp)

    # check if model directory if exists
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # print configuration and save it
    print('Config -----')
    for arg in vars(args):
        print('{0}: {1}'.format(arg, getattr(args, arg)))
    print('------------')

    with open('models/%s/args.txt' %args.name, 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[args.loss]().cuda()

    # create model
    print("=> creating model {}".format(args.arch))
    model = models.__dict__[args.arch](args.in_ch, args.out_ch)

    if torch.cuda.is_available():
        cudnn.benchmark = True
        model = model.cuda()

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise ValueError('optimizer not specified')

    train_loader = RssraiDataLoader(which_set='train', batch_size=args.batch_size, img_size=args.img_size, shuffle=True)
    val_loader = RssraiDataLoader(which_set='val', batch_size=args.batch_size, img_size=args.img_size, shuffle=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'
    ])

    best_loss = 0
    trigger = 0

    for epoch in range(args.epochs):
        print('Epoch [{0:d}/{1:d}]'.format(epoch, args.epochs))

        # train for one epoch
        train_log = train(train_loader=train_loader, model=model, criterion=criterion, optimizer=optimizer)
        # evaluate on validation set
        val_log = validate(val_loader=val_loader, model=model, criterion=criterion)

        print('loss {0:.4f} - iou {1:.4f} - val_loss {2:.4f} - val_iou {3:.4f}'.format(train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            val_log['loss'],
            val_log['iou'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'val_loss', 'val_iou'])

        save_example(model_dir, epoch, model, val_loader)
        log = log.append(tmp, ignore_index=True)
        log.to_csv(os.path.join(model_dir, 'log.csv'), index=False)

        trigger += 1

        if val_log['loss'] > best_loss:
            torch.save(model.state_dict(), os.path.join(model_dir, 'model.pth'))
            best_loss = val_log['loss']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if not args.early_stop is None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break

        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()




