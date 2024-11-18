from argparse import ArgumentParser
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from skimage.measure import regionprops_table
from skimage.filters import threshold_otsu
import tifffile

from nucleocentric.utils.utils import get_row_col_pos
from nucleocentric.utils.io import get_files_in_folder, read_img
from nucleocentric.utils.transforms import get_patch_box
from nucleocentric.features.intensity import get_GT_props


def get_GT_props(img_GT, channels, channel_names, region_names, masks, labels, properties, extra_properties, img_name):
    props = {}
    for ch, ch_name in zip(channels, channel_names):
        for region, mask in zip(region_names,masks):
            props_region = regionprops_table(
                mask,
                img_GT[ch,],
                properties=properties,
                extra_properties=extra_properties
                )
            
            # If label is missing, fill property with NaN
            keys = props_region.keys()
            props[f'{region}_{ch_name}'] = {key: [] for key in props_region.keys() }
            for lbl in labels:
                if lbl in props_region['label']:
                    idx = list(props_region['label']).index(lbl)
                    for key in keys:
                        value = img_name + f'_{lbl:04d}' if key == 'label' else np.around(props_region[key][idx], 2)
                        props[f'{region}_{ch_name}'][key].append(value)
                else:
                    for key in keys:
                        value = img_name + f'_{lbl:04d}' if key == 'label' else np.nan
                        props[f'{region}_{ch_name}'][key].append(value)
    return props


def main(args):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    masks_path = Path(args.masks_path)
    GT_path = Path(args.GT_path)
    files_masks = get_files_in_folder(masks_path, args.file_extension_masks)
    files_GT = get_files_in_folder(GT_path, args.file_extension_GT)
    
    df_GT = pd.DataFrame()
    for (file_masks, file_GT) in tqdm(zip(files_masks, files_GT), total=len(files_masks)):
        if file_masks.stem != file_GT.stem:
            raise ValueError(f"Masks ({file_masks.stem}) and GT ({file_GT.stem}) image names are not corresponding")
        img_name = file_GT.stem

        row, col, pos = get_row_col_pos(img_name)
        print(row, col, pos)
        layout_filepath = args.layout
        culture_layout = pd.read_excel(layout_filepath, sheet_name='Culture', index_col=0)
        target = culture_layout.loc[row, col]

        img_GT = read_img(file_GT)
        masks_cell = read_img(file_masks)

        # Find threshold for for and background
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
            T = threshold_otsu(img_GT[0,][img_GT[0,] != 0]) # looking at != 0 to ignore 0-valued pixels resulting from alignment
            masks_nucleus = np.where(
                img_GT[0,] > T,
                masks_cell,
                0
                )
            masks_cyto = np.where(
                np.logical_and(masks_cell > 0, ~(masks_nucleus>0)),
                masks_cell,
                0
                )
            if args.store_patch:
                filename = str(Path(args.output_path).joinpath(f'{img_name}.tif'))
                tifffile.imwrite(filename, masks_nucleus, dtype=masks_patch.dtype, compression ='zlib')


        # Define properties
        properties=(
            'label',
            'intensity_max',
            'intensity_mean',
            'intensity_min',
            'area',
            'area_convex',
            'area_filled',
            'axis_major_length',
            'axis_minor_length',
            'centroid',
            'eccentricity',
            'equivalent_diameter_area',
            'extent',
            'feret_diameter_max',
            'orientation',
            'perimeter',
            'perimeter_crofton',
            'solidity'
            )

        def intensity_std(region, intensities):
            std = np.std(intensities[region], ddof=1)
            return std if not np.isnan(std) else 0.0
    
        extra_properties=(intensity_std, )


        # Region props inside masks for each channel
        n_channels = img_GT.shape[0]
        labels = np.unique(masks_cell)
        labels = list(labels[labels != 0])
        if args.masked_patch:
            props = get_GT_props(
                img_GT,
                channels=range(0, n_channels),
                channel_names=args.GT_channel_names,
                region_names=('nucleus', 'patch'),
                masks=(masks_nucleus, masks_patch),
                labels=labels,
                properties=properties,
                extra_properties=extra_properties,
                img_name=img_name
                )
        else:
            props = get_GT_props(
                img_GT,
                channels=range(0, n_channels),
                channel_names=args.GT_channel_names,
                region_names=('nucleus', 'cyto', 'cell'),
                masks=(masks_nucleus, masks_cyto, masks_cell),
                labels=labels,
                properties=properties,
                extra_properties=extra_properties,
                img_name=img_name
                )
                
        entry = {}
        for region_ch in props:
            entry.update({f'{region_ch}_{key}': props[region_ch][key] for key in props[region_ch]})
        n = len(labels)
        entry.update({
            'target': [target]*n,
            'true_density': n #added the nr of cells within the mask as a variable --> to correct for true density
        })
        

        df_GT = pd.concat(
            (df_GT, pd.DataFrame.from_dict(entry)),
            axis=0,
            ignore_index=True
            )
                

    # Define GT label columns and predictor columns
    label_columns = [column for column in df_GT.columns if 'label' in column]

    # Check if all rows in df_GT have the same label for each mask
    for label_col in label_columns[1:]:
        if not df_GT[label_columns[0]].equals(df_GT[label_col]):
            raise ValueError(f'Row has different labels for label columns {label_columns[0]} and {label_col}')

    # Create matching column for labels in df_clf and remove old ones
    df_GT['ROI_img_name'] = df_GT[label_columns[0]]
    df_GT = df_GT.drop(columns=label_columns)
     
    GT_name = '-'.join(args.GT_channel_names)
    filename = str(Path(args.output_path).joinpath(f'{timestamp}_GT_data_{GT_name}.csv'))
    df_GT.to_csv(filename, index=False)

   
def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('--GT_path', type=str, required=True,
        help='Path to the GT files')
    parser.add_argument('--GT_channel_names', type=str, nargs= '+', required=True,
        help='Name of GT dataset that will be used to create filename of output')
    parser.add_argument('--masks_path', type=str, required=True,
        help='Path to the masks files')
    parser.add_argument('--layout', type=str, required=True,
        help='Path of the excel file describing the well plate layout')
    parser.add_argument('--file_extension_GT', type=str, default='.tif',
		help='Specify type extension for GT images.')
    parser.add_argument('--file_extension_masks', type=str, default='.tif',
		help='Specify type extension for the masks images.')
    parser.add_argument('--output_path', type=str, required=True,
        help='Output path where results are stored')
    parser.add_argument('--masked_patch', action='store_true',
		help='Select True if the masks given as input are nuclear. This will then create a square patch surrounding the nucleus for feature extraction.')
    parser.add_argument('--store_patch', action='store_true',
	    help='Store the patched masks (only if "masked_patch" is true).')
    parser.add_argument('--patch_size', default=None, type=int,
		help='Patch size used to crop out individual object (if masked_patch is true). If no patch size is defined, a bounding box ' +
			'around each object is used.')

    args = parser.parse_args()

    if isinstance(args.GT_channel_names, str):
        args.GT_channel_names = [args.GT_channel_names]



    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)