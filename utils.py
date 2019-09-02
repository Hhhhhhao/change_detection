import glob
import numpy as np
import torch
import matplotlib.pyplot as plt


def calculate_bce_loss(masks):
    y_minus = 0
    y_plus = 0
    single_total = masks.shape[-1] * masks.shape[-2]
    for m in masks:
        black = len(np.where(m[0]==0)[0])
        white = single_total - black
        y_minus += black
        y_plus += white

    return y_minus / y_plus


def get_pixels(image, boarder_height, boarder_width):
    """
    Get the white pixels from an 2D image, with boarder removed defined by boarder height and boarder width
    :param image: 2D image array
    :param boarder_height: boarder height to be removed
    :param boarder_width: boarder width to be removed
    :return: points
    """
    image_height = image.shape[0]
    image_width = image.shape[1]
    masked_image = image.copy()
    mask = np.zeros_like(image)
    mask[boarder_height:image_height-boarder_height, boarder_width:image_width-boarder_width] = 1
    masked_image[mask == 0] = 0
    points = np.where(masked_image != 0)
    return points


def get_files(directory, format='tif'):
    """
    To get a list of file names in one directory, especially images
    :param directory: a path to the directory of the image files
    :return: a list of all the file names in that directory
    """
    if format is 'png':
        file_list = glob.glob(directory + "*.png")
    elif format is 'tif':
        file_list = glob.glob(directory + "*.tif")
    else:
        raise ValueError("dataset do not support")

    file_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    return file_list


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_example(folder, epoch, model, data_loader):
    """
    save prediction examples during training
    """
    if epoch == 0 or epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            inputs, targets, _ = next(iter(data_loader))
            # if torch.cuda.is_available():
            #     inputs = inputs.cuda()

            preds = model(inputs)
            preds = preds[-1]
            preds = torch.sigmoid(preds)

            if torch.cuda.is_available():
                preds = preds.cpu()
                inputs = inputs.cpu()
                targets = targets.cpu()

            inputs, targets, preds = inputs.numpy(), targets.numpy(), preds.numpy()
            inputs, targets, preds = (inputs * 255).astype('uint8'), (targets * 255).astype('uint8'), (preds * 255).astype('uint8')
            imgs1 = inputs[:, :3, :, :].transpose(0, 2, 3, 1)
            imgs2 = inputs[:, 3:6, :, :].transpose(0, 2, 3, 1)
            targets = targets.transpose(0, 2, 3, 1)
            preds = preds.transpose(0, 2, 3, 1)
            targets = targets[:, :, :, 0]
            preds = preds[:, :, :, 0]

            fig, axs = plt.subplots(imgs1.shape[0], 4)
            for i in range(imgs1.shape[0]):
                if i == 0:
                    axs[i, 0].set_title('I_1')
                    axs[i, 1].set_title('I_2')
                    axs[i, 2].set_title('M')
                    axs[i, 3].set_title('P')
                for j, imgs in enumerate([imgs1, imgs2, targets, preds]):
                    if j <= 1:
                        axs[i, j].imshow(imgs[i])
                    else:
                        axs[i, j].imshow(imgs[i], cmap='gray')
                    axs[i, j].axis('off')

            fig.savefig(folder + '/{}.png'.format(epoch))





