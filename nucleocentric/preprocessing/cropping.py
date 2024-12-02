import numpy as np


def crop_center(img, new_size):
    H, W = img.shape[-2:]
    (new_H, new_W)= new_size
    startx = W//2-(new_W//2)
    starty = H//2-(new_H//2)    
    return img[...,starty:starty+new_H,startx:startx+new_W]


def get_patch_box(centroid, patch_size):
	if isinstance(patch_size, int):
		patch_size = [patch_size] * len(centroid)
	
	row, col = centroid[-2], centroid[-1]
	h_padding = patch_size[-2] / 2
	v_padding = patch_size[-1] / 2
	cmin = (col - h_padding) if h_padding % 1 == 0 else (col - (h_padding+0.5))
	rmin = (row - v_padding) if v_padding % 1 == 0 else (row - (v_padding+0.5))
	cmax = (col + h_padding) if h_padding % 1 == 0 else (col + (h_padding-0.5))
	rmax = (row + v_padding) if v_padding % 1 == 0 else (row + (v_padding-0.5))
	if len(centroid) == 3:
		z = centroid[0]
		d_padding = patch_size[0] / 2
		zmin = (z - d_padding) if d_padding % 1 == 0 else (z - (d_padding+0.5))
		zmax = (z + d_padding) if d_padding % 1 == 0 else (z + (d_padding-0.5))
		box = [int(zmin), int(rmin), int(cmin), int(zmax), int(rmax), int(cmax)]
	else:
		box = [int(rmin), int(cmin), int(rmax), int(cmax)]
	return box


def bbox_crop(img, bbox, padding=None):
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
	d,_,h,w = img.shape
	if padding is None:
		padding = int(0)
	else:
		assert type(padding) is int
	ROI = img[
		max(0, bbox[0]-padding):min(d, bbox[3]+padding),
		:,
		max(0, bbox[1]-padding):min(h, bbox[4]+padding),
		max(0, bbox[2]-padding):min(w, bbox[5]+padding)
		]
	return ROI


def crop_ROI(img, masks, label, bbox, padding=None, masked_patch=False):
	img_ = np.where(masks == label, img, 0) if masked_patch else img
	return bbox_crop(img_, bbox, padding=padding)


