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

from nucleocentric.utils.utils import normalize
from nucleocentric.utils.transforms import unsqueeze_to_ndim


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


def segment_cells(model, input_img, channels, diameter=50, do_3D=False, anisotropy=None, stitch_threshold=0.0):
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
		
	
