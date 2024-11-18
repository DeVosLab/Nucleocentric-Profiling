from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import tifffile
import torch

from nucleocentric.utils.io import get_files_in_folder, read_img
from nucleocentric.utils.transforms import unsqueeze_to_ndim
from nucleocentric.segmentation.segment_nuclei import load_stardist_model, preprocess_img, segment_nuclei


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
	model = load_stardist_model()
	
	# Get all folders to process
	input_path = Path(args.input_path)

	# Segment all images
	files = get_files_in_folder(
		input_path,
		file_extension=args.file_extension
		)

	for file_path in files:
		print(f'file_path: {file_path}')
		img_raw = read_img(file_path)
		if img_raw.ndim == 2:
			img_raw = unsqueeze_to_ndim(img_raw, 4)
		print('Image preprocessing')
		_, img_norm_nuclei = preprocess_img(
			img_raw,
			nuclei_channel_ind=args.nuclei_channel_ind,
			pmin=args.pmin,
			pmax=args.pmax
		)
		masks_nuclei = segment_nuclei(
			model,
			img_norm_nuclei)
		if output_path:
			filename = file_path.stem
			output_path_masks = output_path.joinpath('masks')
			output_path_masks.mkdir(parents=True, exist_ok=True)
			output_path_result_img = output_path.joinpath('result_img')
			output_path_result_img.mkdir(parents=True, exist_ok=True)
			tifffile.imwrite(output_path_masks.joinpath(filename + '_masks_nuclei.tif'), masks_nuclei)

def parse_arguments():
	parser = ArgumentParser()
	parser.add_argument('-i', '--input_path', type=str, required=True, 
		help='Input path to the folder that holds all images')
	parser.add_argument('-o', '--output_path', type=str, required=True, 
		help='Output path were ROIs, masks and result images will be stored')
	parser.add_argument('--file_extension', type=str, default='.nd2',
		help='Specify type extension for the input files.')
	parser.add_argument('--pmin', type=float, default=0.01,
		help='Bottom percentile for input image normalization')
	parser.add_argument('--pmax', type=float, default=99.99,
		help='Upper percentile for image normalization')
	parser.add_argument('--probability', type=float, default=0.6,
		help='Probability threshold for cell detection')
	parser.add_argument('--overlap', type=float, default=0.03,
		help='Overlap treshold allowed for object detection')
	parser.add_argument('--gpu', action='store_true', 
		help='Whether to try to use GPU or not if GPU is available')
	parser.add_argument('--nuclei_channel_ind', type=int, default=0,
		help='Specify the index of the nuclei channel (DAPI)')

	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = parse_arguments()
	main(args)