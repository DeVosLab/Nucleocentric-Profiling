from argparse import ArgumentParser
from calendar import c
import re
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import numpy as np
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift
import tifffile
from matplotlib import pyplot as plt
from util import get_files_in_folder, read_img, normalize

def get_row_col_pos(filename):   ## adjust according to filenames
    print(filename)
    filename = filename.split(".")[0]
    row = filename.split("_")[0][4:5]
    col = filename.split("_")[0][5:7]; col = int(col)
    pos = filename.split("_")[2]; pos = int(pos)
    print(row, col, pos)
    return row, col, pos


def crop_center(img, new_size):
    H, W = img.shape[-2:]
    (new_H, new_W)= new_size
    startx = W//2-(new_W//2)
    starty = H//2-(new_H//2)    
    return img[...,starty:starty+new_H,startx:startx+new_W]


def align_imgs(img_fixed, img_moving, channel2use=0, cval=0, T=None):
    if T is None:
        T, _, _ = phase_cross_correlation(
            img_fixed[channel2use,:,:],
            img_moving[channel2use,:,:],
            normalization='phase'
            )
    T = np.insert(T, 0, 0) # add 0 for channel dimension
    img_moving = shift(img_moving, T, cval=cval)
    return img_moving, T


def align_GT2CP(img_CP, masks, img_GT, channels2use_CP=0, channels2use_GT=0, new_size=(2048,2048), labels=None):
    if labels is None:
        labels = np.unique(masks)
        labels = list(labels[labels != 0])
    else:
        labels_present = np.unique(masks)
        labels_present = list(labels_present[labels_present != 0])
        for lbl in labels_present:
            if lbl not in labels:
                masks = np.where(masks == lbl, 0, masks)

    img_GT_max_channels2use = img_GT[np.r_[channels2use_GT],].max(axis=0)
    img_CP_max_channels2use = img_CP[np.r_[channels2use_CP],].max(axis=0)

    # from matplotlib import pyplot as plt
    # img = np.dstack((
    #     normalize(img_GT_max_channels2use, pmin=1, pmax=99, clip=True),
    #     np.zeros_like(img_CP_max_channels2use),
    #     normalize(img_CP_max_channels2use, pmin=1, pmax=99, clip=True)
    #     ))
    # plt.figure(); plt.imshow(img, cmap='gray')
    # plt.title('Before alignment')
    
    
    image_product = np.fft.fft2(img_GT_max_channels2use) * np.fft.fft2(img_CP_max_channels2use).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product)).real

    ind = np.unravel_index(np.argmax(cc_image, axis=None), cc_image.shape)
    T = np.array(ind) - img_CP.shape[-1]/2

    # Align the images
    img_GT_moved, _ = align_imgs(img_CP, img_GT, T=-T)
    img_GT_max_channels2use_moved = img_GT_moved[np.r_[channels2use_GT],].max(axis=0)

    # Crop out the center "new_size" region
    img_CP = crop_center(img_CP, new_size)
    img_GT_moved = crop_center(img_GT_moved, new_size)
    masks = crop_center(masks, new_size)
    masks = np.squeeze(masks)
    masks = clear_border(masks)

    img_CP_max_channels2use = crop_center(img_CP_max_channels2use, new_size)
    img_GT_max_channels2use_moved = crop_center(img_GT_max_channels2use_moved, new_size)

    # img = np.dstack((
    #     normalize(img_GT_max_channels2use_moved, pmin=1, pmax=99, clip=True),
    #     np.zeros_like(img_CP_max_channels2use),
    #     normalize(img_CP_max_channels2use, pmin=1, pmax=99, clip=True)
    #     ))
    # plt.figure(); plt.imshow(img, cmap='gray')
    # plt.title('After alignment')

    # Identify incomplete masks in img_GT_moved
    missing_GT_data = shift(np.zeros_like(img_GT_max_channels2use), shift=T, cval=1).astype(bool)
    missing_GT_data = crop_center(missing_GT_data, new_size)#[0,]
    
    for lbl in labels[:]:
        binary_lbl = np.where(masks==lbl, True, False)
        if np.logical_and(binary_lbl, missing_GT_data).any():
            masks[binary_lbl] = 0
            labels.remove(lbl) # also remove from labels, because labels is reused later

    # Create binary masks
    binary_mask = np.where(masks > 0, True, False)
    binary_CP = (img_CP_max_channels2use > threshold_otsu(img_CP_max_channels2use)) * binary_mask
    binary_GT = (img_GT_max_channels2use_moved > threshold_otsu(img_GT_max_channels2use)) * binary_mask

    # img = np.dstack((
    #     binary_GT,
    #     np.zeros_like(binary_CP),
    #     binary_CP
    #     )).astype(float)
    # print(img.shape)
    # print(type(img))
    
    # plt.figure(); plt.imshow(img)
    # plt.title('Binary masks')
    # plt.show()
    #return None, None

    # Calculate ratio of DAPI area in CP and GT images
    # for each mask to check if cells are present in both images
    for lbl in labels[:]:
        ratio = binary_GT[masks == lbl].sum()/binary_CP[masks == lbl].sum()
        if ratio < 0.2:
            masks = np.where(masks == lbl, 0, masks)
            labels.remove(lbl)
    return img_GT_moved, labels


def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    
    # Paths to CP and masks image files
    CP_path = args.CP_path
    masks_path = args.masks_path

    # Define the CP and masks output paths
    output_path = Path(args.output_path).joinpath(f'results_{timestamp}')
    output_path_CP = output_path.joinpath('CP')
    output_path_CP.mkdir(exist_ok=True, parents=True)
    output_path_masks = output_path.joinpath('masks')
    output_path_masks.mkdir(exist_ok=True, parents=True)

    # Find all CP and masks files
    CP_files = get_files_in_folder(CP_path, args.file_extension_imgs)
    masks_files = get_files_in_folder(masks_path, args.file_extension_masks)
    GT_files_list = [get_files_in_folder(GT_path, args.file_extension_imgs) for GT_path in args.GT_path]

    print(f'Number of CP files: {len(CP_files)}')
    print(f'Number of masks files: {len(masks_files)}')
    print(f'Number of GT files: {[len(files) for files in GT_files_list]}')

    for i, file_CP in enumerate(tqdm(CP_files,  colour='green')):
        (row, col, pos) = get_row_col_pos(file_CP.name)
        file_masks = next(x for x in masks_files if get_row_col_pos(x.name) == (row, col, pos))
        img_CP = read_img(file_CP)
        img_name = f'{row}-{col:02d}-{pos:02d}'
        print(img_name)
        masks = read_img(file_masks)

        labels = np.unique(masks)
        labels = list(labels[labels != 0])

        for (GT_name, GT_files) in zip(args.GT_name, GT_files_list):
            output_path_GT = output_path.joinpath(f'GT_{GT_name}')
            output_path_GT.mkdir(exist_ok=True, parents=True)

            file_GT = GT_files[i]
            if get_row_col_pos(file_GT.name) != (row, col, pos):
                ValueError(f'GT {get_row_col_pos(file_GT.name)} and CP {(row, col, pos)} files are not responding')

            # Load GT image
            img_GT = read_img(file_GT)

            # Align GT image. Labels are updated with recursion, only labels valid for all GT images are retained
            img_GT_moved, labels = align_GT2CP(
                img_CP,
                masks,
                img_GT,
                channels2use_CP=args.channels2use_CP,
                channels2use_GT=args.channels2use_GT,
                new_size=(2048,2048),
                labels=labels
                )

            # Save moved GT image
            filepath = Path(output_path_GT).joinpath(img_name + '.tif')
            tifffile.imwrite(filepath, img_GT_moved, dtype=img_GT_moved.dtype, compression ='zlib')

        # Save CP image
        new_size =  (2048,2048)
        img_CP = crop_center(img_CP, new_size)
        filepath = Path(output_path_CP).joinpath(img_name + '.tif')
        tifffile.imwrite(filepath, img_CP, dtype=img_CP.dtype, compression ='zlib')

        # Save masks image
        masks = crop_center(masks, new_size)
        masks = np.squeeze(masks)
        masks = clear_border(masks)

        labels_present = np.unique(masks)
        labels_present = list(labels_present[labels_present != 0])
        for lbl in labels_present:
            if lbl not in labels:
                masks = np.where(masks == lbl, 0, masks)
        filepath = Path(output_path_masks).joinpath(img_name + '.tif')
        tifffile.imwrite(filepath, masks, dtype=masks.dtype, compression ='zlib')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--CP_path', type=str, required=True,
        help='Path to the CP files')
    parser.add_argument('--GT_path', type=str, nargs='+', required=True,
        help='Path(s) to the GT files')
    parser.add_argument('--GT_name', type=str, nargs='+', required=True,
        help='Name(s) of the GT datasets') 
    parser.add_argument('--masks_path', type=str, required=True,
        help='Path to the masks files')
    parser.add_argument('--channels2use_CP', nargs='+', type=int, choices=[0,1,2,3,4,5], default=[0,1,2,3,4,5],
		help='Specify the channels to use for training the classifier')  
    parser.add_argument('--channels2use_GT', nargs='+', type=int, choices=[0,1,2,3,4,5], default=[0,1,2,3,4,5],
		help='Specify the channels to use for training the classifier')  
    parser.add_argument('--output_path', type=str, required=True,
        help='Output path where results are stored')
    parser.add_argument('--file_extension_imgs', type=str, default='.nd2',
		help='Specify type extension for the CP and GT images.')
    parser.add_argument('--file_extension_masks', type=str, default='.tif',
		help='Specify type extension for the masks images.')
    args = parser.parse_args()

    args.GT_path = [args.GT_path] if isinstance(args.GT_path, str) else args.GT_path
    args.GT_name = [args.GT_name] if isinstance(args.GT_name, str) else args.GT_name
        
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)