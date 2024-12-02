import numpy as np
from skimage.segmentation import clear_border
from skimage.filters import threshold_otsu
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift

from nucleocentric.preprocessing.cropping import crop_center

def align_imgs(img_fixed, img_moving, channel2use=0, cval=0, T=None):
    if T is None:
        T, _, _ = phase_cross_correlation(
            img_fixed[channel2use,:,:],
            img_moving[channel2use,:,:],
            normalization='phase'
            )
    T = np.insert(T, 0, 0) # add 0 for channel dimension
    img_moving = shift(img_moving, T, cval=cval)
    return img_moving, T


def align_GT2CP(img_CP, masks, img_GT, channels2use_CP=0, channels2use_GT=0, new_size=(2048,2048), labels=None):
    if labels is None:
        labels = np.unique(masks)
        labels = list(labels[labels != 0])
    else:
        labels_present = np.unique(masks)
        labels_present = list(labels_present[labels_present != 0])
        for lbl in labels_present:
            if lbl not in labels:
                masks = np.where(masks == lbl, 0, masks)

    img_GT_max_channels2use = img_GT[np.r_[channels2use_GT],].max(axis=0)
    img_CP_max_channels2use = img_CP[np.r_[channels2use_CP],].max(axis=0)
    
    image_product = np.fft.fft2(img_GT_max_channels2use) * np.fft.fft2(img_CP_max_channels2use).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product)).real

    ind = np.unravel_index(np.argmax(cc_image, axis=None), cc_image.shape)
    T = np.array(ind) - img_CP.shape[-1]/2

    # Align the images
    img_GT_moved, _ = align_imgs(img_CP, img_GT, T=-T)
    img_GT_max_channels2use_moved = img_GT_moved[np.r_[channels2use_GT],].max(axis=0)

    # Crop out the center "new_size" region
    img_CP = crop_center(img_CP, new_size)
    img_GT_moved = crop_center(img_GT_moved, new_size)
    masks = crop_center(masks, new_size)
    masks = np.squeeze(masks)
    masks = clear_border(masks)

    img_CP_max_channels2use = crop_center(img_CP_max_channels2use, new_size)
    img_GT_max_channels2use_moved = crop_center(img_GT_max_channels2use_moved, new_size)

    # Identify incomplete masks in img_GT_moved
    missing_GT_data = shift(np.zeros_like(img_GT_max_channels2use), shift=T, cval=1).astype(bool)
    missing_GT_data = crop_center(missing_GT_data, new_size)#[0,]
    
    for lbl in labels[:]:
        binary_lbl = np.where(masks==lbl, True, False)
        if np.logical_and(binary_lbl, missing_GT_data).any():
            masks[binary_lbl] = 0
            labels.remove(lbl) # also remove from labels, because labels is reused later

    # Create binary masks
    binary_mask = np.where(masks > 0, True, False)
    binary_CP = (img_CP_max_channels2use > threshold_otsu(img_CP_max_channels2use)) * binary_mask
    binary_GT = (img_GT_max_channels2use_moved > threshold_otsu(img_GT_max_channels2use)) * binary_mask

    # Calculate ratio of DAPI area in CP and GT images
    # for each mask to check if cells are present in both images
    for lbl in labels[:]:
        ratio = binary_GT[masks == lbl].sum()/binary_CP[masks == lbl].sum()
        if ratio < 0.2:
            masks = np.where(masks == lbl, 0, masks)
            labels.remove(lbl)

    return img_GT_moved, labels