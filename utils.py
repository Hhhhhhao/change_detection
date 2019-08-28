import glob


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