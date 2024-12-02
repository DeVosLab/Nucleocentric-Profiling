from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import numpy as np
from skimage.segmentation import clear_border
import tifffile
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from nucleocentric import (
    get_files_in_folder,
    read_img,
    get_row_col_pos,
    align_GT2CP,
    crop_center
)

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
        print(f'\n{img_name}')
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