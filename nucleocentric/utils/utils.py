import os
import re
import platform
import random
import numpy as np
import torch
from matplotlib.colors import to_rgb
from skimage.filters import threshold_otsu

# Reproducibility
def set_random_seeds(seed=0):
	if platform.system() == 'Windows':
		os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
	generator = torch.Generator().manual_seed(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.use_deterministic_algorithms(mode=True, warn_only=True)
	return generator
def seed_worker(worker_id):
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)
        

def get_row_col_pos(filename):
    """
    Extracts the row, col and position from the filename of an image in which 
    row and col describes the well and pos the position inside the well where 
    the image was taken

    :param filename: image filename, assumed to be in the form of 
        '{row}-{col}-{pos}',
        with row as a single character and col and pos both as 2-digit numbers
    :type filename: string
    :returns:
        - row - row of the well
        - col - column of the well
        - pos - position inside the well
    """
    row, col, pos = re.match('(\S{1})-(\d{2})-(\d{2})', filename).groups()
    col, pos = int(col), int(pos)
    return row, col, pos



def find_best_z_plane(img):
    ''' Create Z-axis profile and select plane with highest value'''
    # Create Z-axis profile as the sum over all XY channels in function of Z
    z_profile = img.sum(axis=tuple(range(1, img.ndim)))
    # Find slice with the highest overall signal
    idx = z_profile.argmax()
    img = np.expand_dims(img[idx,:,:,:])
    return img


def overlay(img1, img2):
    if img2 is not None:
        T = threshold_otsu(img2)
        img2 = np.dstack((np.zeros(img2.shape), img2, img2))
        color_img = np.where(img2 > T, 1., 0.)
        img = np.dstack((img1,img1,img1)) + 0.5*color_img
        return img
    else:
        return img1


def normalize(x, pmin=1, pmax=99, axis=None, clip=False, eps=1e-20, dtype=np.float32):
	"""Percentile-based image normalization."""
	mi = np.percentile(x,pmin,axis=axis,keepdims=True)
	ma = np.percentile(x,pmax,axis=axis,keepdims=True)
	return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):
	if dtype is not None:
		x   = x.astype(dtype,copy=False)
		mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
		ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
		eps = dtype(eps)

	try:
		import numexpr
		x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
	except ImportError:
		x =                   (x - mi) / ( ma - mi + eps )

	if clip:
		x = np.clip(x,0,1)

	return x


def normalize_minmse(x, target):
	"""Affine rescaling of x, such that the mean squared error to target is minimal."""
	cov = np.cov(x.flatten(),target.flatten())
	alpha = cov[0,1] / (cov[0,0]+1e-10)
	beta = target.mean() - alpha*x.mean()
	return alpha*x + beta


def normalize_mi_ma_images(images, channelwise=True, clip=False):
	c, h, w = images.shape
	if channelwise:
		for i in range(c):
			images[i,:,:] = normalize(images[i,:,:], clip=clip)
	else:
		images = normalize(images, clip=clip)
	return images


def create_composite2D(img, channel_dim, colors=None):
    img_shape = list(img.shape)
    n_channels = img_shape.pop(channel_dim)
    H,W = img_shape
    if colors is None:
        colors = ['blue', 'green', 'magenta', 'gray', 'b', 'r']
        colors = [to_rgb(c) for c in colors]
    else:
        colors = [to_rgb(c) for c in colors]
    n_colors = len(colors)
    if n_channels > n_colors:
        raise RuntimeError('The image has more than 6 channels for which there are default colors. ' +
            'Please provide a color to use for each channels')
    composite_img = np.zeros((H,W,3))
    for c in range(n_channels):
        channel = np.squeeze(img.take((c,), axis=channel_dim))
        channel2rgb = np.dstack((
        colors[c][0]*channel,
        colors[c][1]*channel,
        colors[c][2]*channel
        ))
        composite_img += channel2rgb
    composite_img = normalize(composite_img, clip=True)
    return composite_img





def copy_masks_each_channel(masks, n_channels):
    if n_channels == 1:
        masks_each_channel = masks
    else:
        masks_each_channel = np.repeat( 
            np.expand_dims(masks, 0),
            repeats=n_channels,
            axis=0
        ) 
    return masks_each_channel

