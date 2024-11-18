from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import torch
import tifffile

import sys
sys.path.append(str(Path(__file__).parent.parent))

from nucleocentric import (
	get_files_in_folder,
	read_img,
	unsqueeze_to_ndim,
	max_proj,
	load_cellpose_model,
	preprocess_img_cells,
	segment_cells
)

def main(args):
	time_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
	args.datetime = time_stamp

	# Create the output folder
	output_path = Path(args.output_path)
	output_path.mkdir(exist_ok=True)

	# Store some info on the datset that will be created
	file_name = output_path.joinpath('info.json')
	with open(str(file_name), 'w', encoding='utf-8') as f:
		json.dump(vars(args), f, ensure_ascii=False, indent=4)

	# Set device
	if torch.cuda.is_available():  
		device = torch.device("cuda:0") 
	else:  
		device = torch.device("cpu")
	# Load model
	print('Loading model')
	model = load_cellpose_model(
		target=args.target,
		nuclei_channel_only=args.nuclei_channel_only,
		device=device,
		gpu=args.gpu,
		net_avg=args.net_avg,
		)

	# Get all folders to process
	input_path = Path(args.input_path)

	# Segment all images
	files = get_files_in_folder(
		input_path,
		file_extension=args.file_extension
		)

	for file_path in files:
		print(f'file_path: {file_path}')
		img_raw = read_img(file_path, do_3D=args.do_3D)
		if img_raw.ndim == 2:
			img_raw = unsqueeze_to_ndim(img_raw, 4)
		elif img_raw.ndim == 3:
			if args.do_3D:
				# Add channel dimension: ZxHxW --> Zx1xHxW
				img_raw = np.expand_dims(img_raw, 1)
			else:
				# Add depth dimension: CxHxW --> 1xCxHxW
				img_raw = np.expand_dims(img_raw, 0)
		if args.max_proj:
			filename = file_path.stem
			img_raw = max_proj(img_raw, axis=0, keepdims=True)
			output_path_max = output_path.joinpath('max_proj')
			output_path_max.mkdir(parents=True, exist_ok=True)
			tifffile.imwrite(output_path_max.joinpath(filename + '_max.tif'), img_raw)
		print('Image preprocessing')
		input_img, channels, _, _, _, _ = preprocess_img_cells(
			img_raw,
			nuclei_channel_ind=args.nuclei_channel_ind,
			channels2use=args.channels2use,
			pmin=args.pmin,
			pmax=args.pmax
		)
		print('Cell segmentation')
		masks = segment_cells(
			model,
			input_img,
			channels,
			diameter=args.diameter,
			do_3D=args.do_3D,
			anisotropy=args.anisotropy,
			stitch_threshold=args.stitch_threshold
			)
		print('Cropping ROIs')
		if output_path:
			filename = file_path.stem
			if args.save_masks:
				output_path_masks = output_path.joinpath('masks')
				output_path_masks.mkdir(parents=True, exist_ok=True)
				output_path_result_img = output_path.joinpath('result_img')
				output_path_result_img.mkdir(parents=True, exist_ok=True)
				tifffile.imwrite(output_path_masks.joinpath(filename + '_masks.tif'), masks)


def parse_arguments():
	parser = ArgumentParser()
	parser.add_argument('-i', '--input_path', type=str, required=True, 
		help='Input path to the folder that holds all images')
	parser.add_argument('-o', '--output_path', type=str, required=True, 
		help='Output path were ROIs, masks and result images will be stored')
	parser.add_argument('--file_extension', type=str, default='.tif',
		help='Specify type extension for the input files.')
	parser.add_argument('--save_masks', action='store_true', 
		help='Store masks after segmentation as separate .tif file')
	parser.add_argument('--save_raw', action='store_true',
		help='Save ROIs cropped from raw image instead of normalized image.')
	parser.add_argument('--max_proj', action='store_true',
		help='Perform maximum projection over all Z-plannes (axis=0)')
	parser.add_argument('--pmin', type=float, default=0.01,
		help='Bottom percentile for input image normalization')
	parser.add_argument('--pmax', type=float, default=99.99,
		help='Upper percentile for image normalization')
	parser.add_argument('--do_3D', action='store_true',
		help='Perform segmentation in 3D. The orders of dimensions is supposed to be ZxCxHxW.')
	parser.add_argument('--stitch_threshold', type=float, default=0.,
		help='Perform segmentation in 3D by joining masks in consecutive slices if IoU > stitching threshold.')
	parser.add_argument('--anisotropy', type=float, default=None,
		help='Optional rescaling factor (e.g. set to 2.0 if Z is sampled half as dense as X or Y)')
	parser.add_argument('--gpu', action='store_true', 
		help='Whether to try to use GPU or not if GPU is available')
	parser.add_argument('--net_avg', action='store_true',
		help='Whether to use average prediction of four models, or of single model')
	parser.add_argument('--diameter', type=int, default=None,
		help='Estimated diameter of cells present in the image')
	parser.add_argument('--omni', action='store_true',
		help='Use omnipose model instead of cellpose.')
	parser.add_argument('--channels2use', nargs='+', type=int, choices=[0,1,2,3,4,5,6,7,8], default=[0,1,2,3,4,5],
		help='Specify the channels to use for segmentation')
	parser.add_argument('--channels2store', nargs='+', type=int, choices=[0,1,2,3,4,5,6,7,8], default=[0,1,2,3,4,5],
		help='Specify the channels to store after segmentation. Be aware that only information ' +
			'inside each mask is stored.')
	parser.add_argument('--nuclei_channel_ind', type=int, default=0,
		help='Specify the index of the nuclei channel (DAPI)')
	parser.add_argument('--target', default='cyto', choices=['nuclei', 'cyto'],
		help='Choose if nuclei or cytoplasm is the target for segmentation. ' +
			'Defaults to nuclei if only the args.nuclei_channel_id is listed in argrs.channels2use')
	parser.add_argument('--masked_patch', action='store_true',
		help='Set background patch to zeros based on mask.')
	parser.add_argument('--patch_size', default=None, type=int,
		help='Patch size used to crop out individual object. If no patch size is defined, a bounding box ' +
			'around each object is used.')

	args = parser.parse_args()

	if args.max_proj and args.do_3D:
		raise RuntimeError('"--max_proj" cannot be used for 3D segmenation ("--do_3D").')

	args.channels2use = list(args.channels2use) if isinstance(args.channels2use, int) else args.channels2use
	args.channels2store = list(args.channels2store) if isinstance(args.channels2store, int) else args.channels2store

	args.nuclei_channel_only = True if (len(args.channels2use) == 1 and args.nuclei_channel_ind in args.channels2use) else False
	if args.target == 'cyto' and args.nuclei_channel_only:
		raise RuntimeError(f'Target for segmentation (args.target) cannot be "{args.target}"' +
			f' if args.channels2use ({args.channels2use}) only containts the nuclei channel ({args.nuclei_channel_ind}).')
	
	return args


if __name__ == '__main__':
	args = parse_arguments()
	main(args)