import sys

path = r'cellpose'

if path not in sys.path:
	sys.path.append(path)

import os
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
import json

import numpy as np
import skimage
import tifffile
import torch
from matplotlib import pyplot as plt
from skimage.filters import median
from skimage.measure import regionprops_table
from skimage.morphology import disk, ball
from skimage.segmentation import clear_border

from cellpose import models
from util import (create_composite2D, get_files_in_folder, normalize, 
					max_proj, read_img, unsqueeze_to_ndim, create_composite2D)

def get_patch_box(centroid, patch_size):
	z, row, col = centroid
	d_padding = patch_size / 2    
	h_padding = patch_size / 2
	v_padding = patch_size / 2
	zmin = (z - d_padding) if d_padding % 1 == 0 else (z - (d_padding+0.5))
	cmin = (col - h_padding) if h_padding % 1 == 0 else (col - (h_padding+0.5))
	rmin = (row - v_padding) if v_padding % 1 == 0 else (row - (v_padding+0.5))
	zmax = (z + d_padding) if d_padding % 1 == 0 else (z + (d_padding-0.5))
	cmax = (col + h_padding) if h_padding % 1 == 0 else (col + (h_padding-0.5))
	rmax = (row + v_padding) if v_padding % 1 == 0 else (row + (v_padding-0.5))
	box = [int(zmin), int(rmin), int(cmin), int(zmax), int(rmax), int(cmax)]
	return box

def bbox_crop(masked_img, bbox, padding=None):
	'''
	Function to crop the volume inside a 2D boudning box
	params:
	- img (2D numpy array): input img
	- bbox (list): bounding box
	- padding (int or None): number of 0 pixels to pad the cropped volume with.
		No padding is used if padding is None.
	returns:
	- ROI: cropped out 2D region of interest inside the bounding box in the input volume.
	'''
	d,_,h,w = masked_img.shape
	if padding is None:
		padding = int(0)
	else:
		assert type(padding) is int
	ROI = masked_img[
		max(0, bbox[0]-padding):min(d, bbox[3]+padding),
		:,
		max(0, bbox[1]-padding):min(h, bbox[4]+padding),
		max(0, bbox[2]-padding):min(w, bbox[5]+padding)
		]
	return ROI


def crop_ROI(img, masks, label, bbox, padding=None, masked_patch=False):
	img_ = np.where(masks == label, img, 0) if masked_patch else img
	return bbox_crop(img_, bbox, padding=padding)


def crop_bboxs(img, masks, labels, bboxs, centroids, padding=None, mask_padding=None,
			masked_patch=False, output_path=None, prefix=None, normalize_grayscale=False,
			save_raw=False):
	'''
	Function to crop all individual masks in a volume
	params:
	img (2D numpy array): input gray scale image
	bboxs (list): list of bbox for each mask
	returns:
	ROIs (list): list of cropped out ROIs, one for each mask
	'''
	ROIs=[]
	for label, bbox, centroid in zip(labels, bboxs, centroids):
		ROI = crop_ROI(img, masks, label, bbox, padding=padding, masked_patch=masked_patch)
		if output_path is not None:
			save_ROI(
				ROI, centroid, output_path, prefix=prefix, mask_padding=mask_padding,
				normalize_grayscale=normalize_grayscale, save_raw=save_raw)
		else:
			ROIs.append(ROI)
			return ROIs


def create_ROI_name(centroid, mask_padding=None, zero_padding=4):
	'''
	Function to create a name for a ROI according to its centroid coordinates
	params:
	- centroid: centroid for cropped out volume
	- zero_padding (int): number of zeros with which the coordinate values are maximally padded
	returns:
	ROI_name (str): name of the VOI according to its centroid
	'''
	if mask_padding is not None:
		centroid = centroid - [pad[0] for pad in  mask_padding]
	ROI_name = 'coords_Z' + str(centroid[0]).zfill(zero_padding) + \
		'-Y' + str(centroid[1]).zfill(zero_padding) + \
		'-X' + str(centroid[2]).zfill(zero_padding)
	return ROI_name


def save_ROI(ROI, centroid, output_path, prefix=None, mask_padding=None, normalize_grayscale=True, save_raw=False):
	'''
	Function to save slices of individual ROIs in separate folders
	params:
	- ROI: cropped out volume
	- centroids: centroid of the cropped out volume
	- output_path: path in which the slices of the individual ROIs will be saved
	- prefix (str): prefix to add to all file names
	- normalize_grayscale (bool): boolean to min-max normalize ROI grayscale values to 
		[0, 1] (=default) or not
	'''
	if prefix is None:
		prefix = ''
	file_name = os.path.join(output_path, prefix + create_ROI_name(centroid, mask_padding)+'.tif')

	if save_raw:
		tifffile.imwrite(file_name, ROI, dtype=ROI.dtype)
	else:
		if normalize_grayscale:
			ROI = (ROI - ROI.min())/(ROI.max() - ROI.min())
		tifffile.imsave(file_name, (255*ROI).astype(np.uint8))

def load_cellpose_model(target, nuclei_channel_only, device, gpu=False, net_avg=False):
	model_type = 'cyto2' if target == 'cyto' and not nuclei_channel_only else 'nuclei'
	return models.Cellpose(gpu=gpu, model_type=model_type, device=device, net_avg=net_avg)

def preprocess_img(img_raw, nuclei_channel_ind=0, channels2use=[0,1,2,3,4,5], pmin=0.01, pmax=99.99):
	print(f'	Raw image shape: {img_raw.shape}')
	d, n_channels, _, _ = img_raw.shape
	channel_dim = 1
	img_norm = np.zeros(img_raw.shape, dtype=float) # (C, H, W)
	img_med = np.zeros(img_raw.shape, dtype=float)
	for c in range(n_channels):
		channel_raw = img_raw[:,c,:,:]
		channel_norm = normalize(channel_raw, pmin=pmin, pmax=pmax, clip=True)
		img_norm[:,c,:,:] = channel_norm

		se = np.expand_dims(disk(3), axis=0) if d == 1 else ball(1)
		channel_med = median(channel_raw, se)
		channel_med = normalize(channel_med, pmin=pmin, pmax=pmax, clip=True)     
		img_med[:,c,:,:] = channel_med

	if nuclei_channel_ind in channels2use:
		img_nuclei = img_med[:,nuclei_channel_ind,:,:]
		if len(channels2use) == 1:
			input_img = np.expand_dims(img_nuclei, channel_dim) # (Z, 1, H, W)
			channels = [0,0]
			img_cyto = None
		else:
			channels2use = [c for c in channels2use if c != nuclei_channel_ind]
			img_cyto = img_med[:,np.r_[channels2use],:,:].mean(axis=channel_dim, keepdims=True) # (Z,H,W)
			input_img = np.concatenate(
				(img_cyto,    							   # (Z, 1, H, W)
				np.expand_dims(img_nuclei, channel_dim)),  # (Z, 1, H, W) 
				axis=channel_dim
			)											   # --> (Z, 2, H, W)
			channels = [0,1]
	else:
		img_nuclei = None
		input_img = img_cyto = img_med[:,np.r_[channels2use],:,:].mean(axis=channel_dim, keepdims=True) # (Z,1,H,W)
		channels = [0,0]
	return input_img, channels, img_norm, img_med, img_nuclei, img_cyto


def segment_img(model, input_img, channels, diameter=50, do_3D=False, anisotropy=None, stitch_threshold=0.0):
	if not do_3D:
		input_img = np.squeeze(input_img, axis=0)
	print(f'	Input image shape: {input_img.shape}')
	# Do prediction
	if stitch_threshold >0.:
		do_3D = False
	masks, _, _, _ = model.eval(
		input_img,
		channels=channels,
		normalize=False,    # already normalized
		diameter=diameter,
		do_3D=do_3D,
		anisotropy=anisotropy,
		stitch_threshold=stitch_threshold
		)
	# Remove masks connected to the border
	masks = clear_border(masks) # (H,W) or (Z,H,W)
	masks = unsqueeze_to_ndim(masks, n_dim=3) # (Z,H,W)
	return masks


def get_ROIs_from_masks(img, masks, patch_size=None, masked_patch=False, channels2store=[0,1,2,3,4,5],
						output_path=None, prefix=None, normalize_grayscale=False, save_raw=False):
	d = img.shape[0]
	# Make sure patches fit the image
	if patch_size:
		d_padding = (patch_size//2,patch_size//2) if d > 1 else (0,0) # depthwise padding
		mask_padding = (d_padding, (patch_size//2,patch_size//2), (patch_size//2,patch_size//2))
		img_padding = (d_padding, (0,0), (patch_size//2,patch_size//2), (patch_size//2,patch_size//2))
		masks = np.pad(masks, mask_padding)
		img = np.pad(img, img_padding)
	
	# Get properties
	props = regionprops_table(
		masks, 
		properties=('label', 'bbox', 'centroid', 'equivalent_diameter_area')
		)
	labels = props['label']
	diameters = props['equivalent_diameter_area']
	centroids = [np.round([z, row, col]).astype(int) for z, row, col in \
		zip(props['centroid-0'],props['centroid-1'],props['centroid-2'])]
	if patch_size is not None:
		bboxs = [get_patch_box(centroid, patch_size) for centroid in centroids]
	else:
		bboxs = [np.round([zmin, ymin, xmin, zmax, ymax, xmax]).astype(int) for zmin, ymin, xmin, zmax, ymax, xmax in \
			zip(props['bbox-0'],props['bbox-1'],props['bbox-2'],props['bbox-3'], props['bbox-4'],props['bbox-5'])]

	# Store ROIs identified by each mask
	n_channels = len(channels2store)
	channel_dim = 1
	img = img[:,np.r_[channels2store],:,:]
	if n_channels == 1:
		masks_each_channel = np.expand_dims(masks, channel_dim)
	else:
		masks_each_channel = np.repeat( # copy mask for each channel
			np.expand_dims(masks, channel_dim),
			repeats=n_channels,
			axis=channel_dim
		)
	patch_padding = 0 if patch_size else 5
	ROIs = crop_bboxs(
		img,
		masks_each_channel,
		labels,
		bboxs,
		centroids,
		padding=patch_padding,
		mask_padding=mask_padding,
		masked_patch=masked_patch,
		output_path=output_path, prefix=prefix, normalize_grayscale=normalize_grayscale,
		save_raw=save_raw
		)
	return ROIs, centroids, bboxs, diameters

def save_result_img(img_med, masks, output_path, file_path, channels2use, diameters=None, z=None):
	# Take 2D slice of 3D image, or z=0 for 2D image
	if z is None:
		# Take the Z-slice in the middle of the stack to greate figure with
		z = int(img_med.shape[0]/2)
	img_med = img_med[z,]
	masks = masks[z,]
	channel_dim = 0 # from 1 to 0 since z dimension is gone

	# Create RGB composite image of the input
	img_composite = create_composite2D(img_med[np.r_[channels2use],:,:], channel_dim=channel_dim)
	# Create RGB overlay image of masks
	img_mean = img_med[np.r_[channels2use],:,:].mean(axis=channel_dim) # (H,W)
	output = skimage.color.label2rgb(masks, image=img_mean, kind='overlay')

	n_subplots = 3 if diameters is not None else 2
	# Input image
	fig = plt.figure(figsize=(16,9))
	ax = plt.subplot(1,n_subplots,1)
	ax.imshow(np.squeeze(img_composite))
	plt.axis('off')
	plt.title(f'Input')

	# Output (overlay of masks on top of input)
	ax = plt.subplot(1,n_subplots,2)
	ax.imshow(output)
	plt.axis('off')
	plt.title(f'Prediction')

	# And boxplot of the diameters of the segmented labels
	if n_subplots == 3:
		ax = plt.subplot(1,n_subplots,3)
		plt.boxplot(diameters)
		plt.title(f'Equivalent cell diamter') # (px size: {pixel_size:.2f} \u03BCm)')
		plt.ylabel('Equivalent cell diamater (pixels)') #(\u03BCm)')
		plt.ylim((0, 100))
		plt.grid('on')
		plt.tight_layout()
	
	# Save figure
	file_name = output_path.joinpath(file_path.stem + '.png')
	fig.savefig(file_name)
	plt.close()
	return
		
	
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
		#omni=args.omni
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
		input_img, channels, img_norm, img_med, img_nuclei, _ = preprocess_img(
			img_raw,
			nuclei_channel_ind=args.nuclei_channel_ind,
			channels2use=args.channels2use,
			pmin=args.pmin,
			pmax=args.pmax
		)
		print('Cell segmentation')
		masks = segment_img(
			model,
			input_img,
			channels,
			diameter=args.diameter,
			do_3D=args.do_3D,
			anisotropy=args.anisotropy,
			stitch_threshold=args.stitch_threshold
			)
		print('Cropping ROIs')
		diameters=None
		if output_path:
			filename = file_path.stem
			if args.save_masks:
				output_path_masks = output_path.joinpath('masks')
				output_path_masks.mkdir(parents=True, exist_ok=True)
				output_path_result_img = output_path.joinpath('result_img')
				output_path_result_img.mkdir(parents=True, exist_ok=True)
				tifffile.imwrite(output_path_masks.joinpath(filename + '_masks.tif'), masks)
			if args.save_ROIs:
				output_path_ROI = output_path.joinpath('ROI')
				output_path_ROI.mkdir(exist_ok=True)
				output_path_ROI_img = output_path_ROI.joinpath(filename)
				output_path_ROI_img.mkdir(parents=True, exist_ok=True)
				_, _, _, diameters = get_ROIs_from_masks(
					img_raw if args.save_raw else img_norm, # crop ROIs from raw or normalized image
					masks,
					patch_size=args.patch_size,
					masked_patch=args.masked_patch,
					channels2store=args.channels2store,
					output_path=output_path_ROI_img	,
					normalize_grayscale=False,
					save_raw=args.save_raw
					)  
			save_result_img(img_med, masks, output_path_result_img, file_path, 
				channels2use=args.channels2use, diameters=diameters)

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
	parser.add_argument('--save_ROIs', action='store_true', 
		help='Store ROIs after segmentation as separate .tif file')
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
	
	# if args.do_3D and args.stitch_threshold > 0.:
	# 	raise RuntimeError(f'args.do3D cannot be {args.do_3D} if args.stitch_threshold > 0.' +
	# 		f'Should only one of these methods to obtain a 3D image')
	return args


if __name__ == '__main__':
	args = parse_arguments()
	main(args)