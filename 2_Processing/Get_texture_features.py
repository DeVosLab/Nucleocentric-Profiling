### Texture extract from nuclear region crops + patches surrounding the nuclear crops

from argparse import ArgumentParser
import re
from tqdm import tqdm
from pathlib import Path
import numpy as np
from skimage.measure import regionprops_table
import tifffile
import skimage.feature as feature
import pandas as pd
from skimage.filters import threshold_otsu
from datetime import datetime
from matplotlib import pyplot as plt
import scipy

from util import get_files_in_folder, read_img, unsqueeze_to_ndim
from Segment_cells import get_patch_box, crop_ROI

def get_row_col_pos(filename):
    row, col, pos = re.match('(\S{1})-(\d{2})-(\d{2})', filename).groups()
    col, pos = int(col), int(pos)
    return row, col, pos

def main(args):
     # Paths to CP and masks a image files
    CP_path = args.CP_path
    masks_path = args.masks_path

    # Define the output paths
    output_path = Path(args.output_path)

    # Find all CP, masks and GT files
    CP_files = get_files_in_folder(CP_path, args.file_extension_imgs)
    masks_files = get_files_in_folder(masks_path, args.file_extension_masks)
    df = pd.DataFrame(columns = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM', 'filename','channel', 'region'])

    for (file_CP, file_masks) in tqdm(zip(CP_files, masks_files), total=len(CP_files)):
        img_name = file_CP.stem
        if img_name != file_masks.stem:
            raise ValueError(f"CP ({img_name}) and masks ({file_masks.stem}) image names are not corresponding")

        img_CP = read_img(file_CP)
        print(file_CP)
        masks_cell = read_img(file_masks)
        if args.masked_patch:
            masks_nucleus = masks_cell
            props = regionprops_table(
                masks_nucleus, 
                properties=('label', 'bbox', 'centroid', 'equivalent_diameter_area')
                )
            labels = props['label']
            diameters = props['equivalent_diameter_area']
            centroids = [np.round([row, col]).astype(int) for row, col in \
                zip(props['centroid-0'],props['centroid-1'])]
            masks_patch = np.zeros((2048, 2048),dtype=np.uint16) # initialize mask
            bboxs = [get_patch_box(centroid, args.patch_size) for centroid in centroids]
            for idx, box in enumerate(bboxs): 
                rmin = box[0]
                cmin = box[1]
                rmax = box[2]
                cmax = box[3]
                masks_patch[rmin:rmax, cmin:cmax] = idx+1
            if args.store_patch:
                filename = str(Path(args.output_path).joinpath(f'{img_name}.tif'))
                tifffile.imwrite(filename, masks_patch, dtype=masks_patch.dtype, compression ='zlib')
        else:
            T = threshold_otsu(img_CP[0,][img_CP[0,] != 0]) # looking at != 0 to ignore 0-valued pixels resulting from alignment
            masks_nucleus = np.where(
                img_CP[0,] > T,
                masks_cell,
                0
                )
            if args.store_patch:
                filename = str(Path(args.output_path).joinpath(f'{img_name}.tif'))
                tifffile.imwrite(filename, masks_nucleus, dtype=masks_patch.dtype, compression ='zlib')

        masks_list = [masks_nucleus]

        for mask in masks_list:
            img_CP = unsqueeze_to_ndim(img_CP, 4)
            masks = unsqueeze_to_ndim(mask, 3)
            if (mask == masks_nucleus).all():
                name = 'nucleus'
            else:
                print('error')
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
                for ch in range(n_channels):
                    img = ROI[0,ch,:,:].astype('int') ## loop over channels!!
                    max_value = np.max(img)
                    img_norm = (img/max_value)*255
                    img_norm = img_norm.astype('int')
                    try:
                        graycom = feature.graycomatrix(img_norm, [1], [0], levels=256)  # loop over angles np.pi/4, np.pi/2, 3*np.pi/4
                        # Find the GLCM properties
                        contrast = float(feature.graycoprops(graycom, 'contrast'))
                        dissimilarity = float(feature.graycoprops(graycom, 'dissimilarity'))
                        homogeneity = float(feature.graycoprops(graycom, 'homogeneity'))
                        energy = float(feature.graycoprops(graycom, 'energy'))
                        correlation = float(feature.graycoprops(graycom, 'correlation'))
                        ASM = float(feature.graycoprops(graycom, 'ASM'))
                        tx = [contrast, dissimilarity, homogeneity, energy, correlation, ASM, filename, ch, name]
                        df.loc[len(df)] = tx
                    except:
                        pass # doing nothing on exception
                
    
    # rearrange dataframe if necessary
    # region_df_cell = df.loc[df['region'] == 'patch']; region_df_cell = region_df_cell.drop(['region'], axis = 1); colnames = ['contrast_patch', 'dissimilarity_patch', 'homogeneity_patch', 'energy_patch', 'correlation_patch', 'ASM_patch', 'filename','channel']; region_df_cell.set_axis(colnames, axis=1, inplace=True)
    # region_df_nucleus = df.loc[df['region'] == 'nucleus'] ; region_df_nucleus = region_df_nucleus.drop(['region'], axis = 1); colnames = ['contrast_nucleus', 'dissimilarity_nucleus', 'homogeneity_nucleus', 'energy_nucleus', 'correlation_nucleus', 'ASM_nucleus', 'filename','channel']; region_df_nucleus.set_axis(colnames, axis=1, inplace=True)
    # df_merged = pd.merge(region_df_cell, region_df_nucleus, on=['filename','channel'])
    # df = df.drop(['region'], axis=1)
    # channel_df_0 = df.loc[df['channel'] == 0]; channel_df_0 = channel_df_0.drop(['channel'], axis = 1); colnames = ['contrast_nucleus_DAPI', 'dissimilarity_nucleus_DAPI', 'homogeneity_nucleus_DAPI', 'energy_nucleus_DAPI', 'correlation_nucleus_DAPI', 'ASM_nucleus_DAPI','filename']; channel_df_0.set_axis(colnames, axis=1, inplace=True)
    # channel_df_1 = df.loc[df['channel'] == 1]; channel_df_1 = channel_df_1.drop(['channel'], axis = 1); colnames = ['contrast_nucleus_FITC', 'dissimilarity_nucleus_FITC', 'homogeneity_nucleus_FITC', 'energy_nucleus_FITC', 'correlation_nucleus_FITC', 'ASM_nucleus_FITC','filename']; channel_df_1.set_axis(colnames, axis=1, inplace=True)
    # channel_df_2 = df.loc[df['channel'] == 2]; channel_df_2 = channel_df_2.drop(['channel'], axis = 1); colnames = ['contrast_nucleus_Cy3', 'dissimilarity_nucleus_Cy3', 'homogeneity_nucleus_Cy3', 'energy_nucleus_Cy3', 'correlation_nucleus_Cy3', 'ASM_nucleus_Cy3','filename']; channel_df_2.set_axis(colnames, axis=1, inplace=True)
    # channel_df_3 = df.loc[df['channel'] == 3]; channel_df_3 = channel_df_3.drop(['channel'], axis = 1); colnames = ['contrast_nucleus_Cy5', 'dissimilarity_nucleus_Cy5', 'homogeneity_nucleus_Cy5', 'energy_nucleus_Cy5', 'correlation_nucleus_Cy5', 'ASM_nucleus_Cy5','filename']; channel_df_3.set_axis(colnames, axis=1, inplace=True)
    # df_merged = pd.merge(channel_df_0, channel_df_1, on=['filename'])
    # df_merged = pd.merge(df_merged, channel_df_2, on=['filename'])
    # df_merged = pd.merge(df_merged, channel_df_3, on=['filename'])
    # df = df_merged.copy()
    # print(df_merged)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = str(Path(args.output_path).joinpath(f'{timestamp}_texture.csv'))
    df.to_csv(filename, index=False)


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
		help='Select True if the masks given as input are nuclear. This will then create a square patch surrounding the nucleus for feature extraction.')
    parser.add_argument('--store_patch', action='store_true',
	    help='Store the patched masks (only if "masked_patch" is true).')
    parser.add_argument('--patch_size', default=None, type=int,
		help='Patch size used to crop out individual object (if masked_patch is true). If no patch size is defined, a bounding box ' +
			'around each object is used.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    main(args)


