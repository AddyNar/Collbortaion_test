import os
import glob
import tensorflow.keras.backend as K


def get_image_pair_fnames(base_dir, dstype):
    """
    Return image-mask pairs in a glob style path.

    Input arguments:
    base_dir -- Base directory where image splits are stored.
    dstype -- Data split to work on; usually val, train or split.

    Returns:
    fname_pairs -- List of tuples; each tuple ordered as (mask,image)
    """
    # Get the folders where the images and labels are stored
    base_label_dir = os.path.join(base_dir, 'gtFine')
    base_image_dir = os.path.join(base_dir, 'leftImg8bit')

    # Define the base directory and the directory for fine labels and images
    base_dstype_label_dir = os.path.join(base_label_dir, dstype)
    base_dstype_image_dir = os.path.join(base_image_dir, dstype)
    print('The place where '+dstype+' labels are at:'+base_dstype_label_dir)
    print('The place where '+dstype+' images are at:'+base_dstype_image_dir)

    # Get the folder names from which the images will be taken from
    folder_names = os.listdir(base_dstype_image_dir)
    print(folder_names)

    # Define empty filename lists for both labels and images
    # Note that these are saved as full paths
    fname_pairs = []
    # Now we need to go over each of them and get the pairs
    for folder_name in folder_names:
        print("Currently accesing in both images and labels:"+folder_name)

        # Access the folder in both the image and labe sets
        fname_path_label = os.path.join(base_dstype_label_dir, folder_name)

        # Now we access the relevant files from images and labels
        fname_label = glob.glob(os.path.join(fname_path_label, '*gtFine_labelIds.png'))
        fname_image = [labelname.replace('gtFine_labelIds.png', 'leftImg8bit.png').replace('gtFine', 'leftImg8bit') for labelname in fname_label]

        for i in range(len(fname_label)):
            fname_pairs.append([fname_label[i], fname_image[i]])

        print(len(fname_label))
        print(len(fname_image))
        print(len(fname_pairs))

    return fname_pairs


def iou_coef(y_true, y_pred, smooth=1):
    """
    Take ground truth and prediction mask and return the IoU score.

    Input Arguments:
    y_true -- The hot-one encoded mask generated from the Label ID's folder
    y_pred -- The output prediction of the CNN of the same dimensions as y_true
    smooth -- The smoothness value of the computation; default = 1

    Returns:
    iou -- The IoU score
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3])+K.sum(y_pred, [1, 2, 3])-intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def dice_coef(y_true, y_pred, smooth=1):
    """
    Take ground truth and prediction mask and return the dice coefficient.

    Input Arguments:
    y_true -- The hot-one encoded mask generated from the Label ID's folder
    y_pred -- The output prediction of the CNN of the same dimensions as y_true
    smooth -- The smoothness value of the computation; default = 1

    Returns:
    dice -- The Dice Coefficient
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth)/(union + smooth), axis=0)
    return dice
