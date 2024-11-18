from argparse import ArgumentParser
from pathlib import Path
import numpy as np
from skimage.measure import regionprops_table
import tifffile
from tqdm import tqdm

from nucleocentric import (
    get_files_in_folder,
    read_img,
    unsqueeze_to_ndim,
    get_patch_box,
    crop_ROI
)

def main(args):
     # Paths to CP and masks a image files
    CP_path = args.CP_path
    masks_path = args.masks_path

    # Define the output paths
    output_path = Path(args.output_path)

    # Find all CP, masks and GT files
    CP_files = get_files_in_folder(CP_path, args.file_extension_imgs)
    masks_files = get_files_in_folder(masks_path, args.file_extension_masks)

    for (file_CP, file_masks) in tqdm(zip(CP_files, masks_files), total=len(CP_files)):
        img_name = file_CP.stem
        print(img_name)
        if img_name != file_masks.stem:
            raise ValueError(f"CP ({img_name}) and masks ({file_masks.stem}) image names are not corresponding")

        img_CP = read_img(file_CP)
        masks = read_img(file_masks)
        masks = masks.astype(int)

        img_CP = unsqueeze_to_ndim(img_CP, 4)
        masks = unsqueeze_to_ndim(masks, 3)
        props = regionprops_table(
            masks, 
            properties=('label', 'bbox', 'centroid')
        )
        labels = props['label']
        centroids = [np.round([z, row, col]).astype(int) for z, row, col in \
            zip(props['centroid-0'],props['centroid-1'],props['centroid-2'])]
        bboxs = [get_patch_box(centroid, patch_size=args.patch_size) for centroid in centroids]

        # Save algined CP ROIs in subfolder for each image
        for lbl, bbox in zip(labels, bboxs):
            filename = img_name + f'_{lbl:04d}.tif'
            # CP ROI
            channel_dim = 1
            n_channels = img_CP.shape[channel_dim]
            masks_each_channel = np.repeat( # copy mask for each channel
                np.expand_dims(masks, channel_dim),
                repeats=n_channels,
                axis=channel_dim
            )
            ROI = crop_ROI(img_CP, masks_each_channel, lbl, bbox, masked_patch=args.masked_patch)
            filepath = Path(output_path).joinpath(img_name, filename)
            filepath.parent.mkdir(exist_ok=True, parents=True)
            tifffile.imwrite(filepath, ROI, dtype=ROI.dtype, compression ='zlib')


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--CP_path', type=str, required=True,
        help='Path to the CP files')
    parser.add_argument('--masks_path', type=str, required=True,
        help='Path to the masks files')
    parser.add_argument('--output_path', type=str, required=True,
        help='Output path where results are stored')
    parser.add_argument('--file_extension_imgs', type=str, default='.tif',
		help='Specify type extension for the CP and GT images.')
    parser.add_argument('--file_extension_masks', type=str, default='.tif',
		help='Specify type extension for the masks images.')
    parser.add_argument('--masked_patch', action='store_true',
		help='Set background patch to zeros based on mask.')
    parser.add_argument('--patch_size', type=int, default=192,
        help='Size of the patch cropped around each label/ROI')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)