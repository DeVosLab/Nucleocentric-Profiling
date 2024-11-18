from stardist.models import StarDist2D
from csbdeep.utils import normalize
import matplotlib.pyplot as plt
from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
import skimage
import tifffile
import torch
from matplotlib import pyplot as plt
from skimage.segmentation import clear_border
from src.util import (get_files_in_folder, normalize, read_img, unsqueeze_to_ndim)


def load_stardist_model(name='2D_versatile_fluo'):
	return StarDist2D.from_pretrained(name)

def preprocess_img(img_raw, nuclei_channel_ind=0, pmin=0.01, pmax=99.99):
	print(f'	Raw image shape: {img_raw.shape}')
	img_nuclei = img_raw[nuclei_channel_ind,:,:]
	img_norm_nuclei = np.zeros(img_nuclei.shape, dtype=float) # (C, H, W)
	img_norm_nuclei = normalize(img_nuclei, pmin=pmin, pmax=pmax, clip=True)
	return img_nuclei, img_norm_nuclei

def segment_nuclei(model, img_nuclei):
    masks_nuclei,_ = model.predict_instances(img_nuclei,  prob_thresh = args.probability ,nms_thresh = args.overlap)
    masks_nuclei = clear_border(masks_nuclei) # (H,W) or (Z,H,W)
    masks_nuclei = unsqueeze_to_ndim(masks_nuclei, n_dim=3) # (Z,H,W)
    return masks_nuclei