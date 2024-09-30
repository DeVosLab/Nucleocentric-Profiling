import os
import re
import platform
import json
import random
import numbers
from pathlib import Path

import numpy as np
import tifffile
import torch
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as F
from torchvision.datasets import ImageFolder
from nd2reader import ND2Reader
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from matplotlib.colors import to_rgb
from skimage.filters import threshold_otsu
from sklearn.model_selection import StratifiedGroupKFold

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
        
# IO
def load_custom_config(path='custom_config.json'):
	with open(path) as f:
		custom_config = json.load(f)
	return custom_config

def check_extenstion(filename, extension='.tif'):
    if not isinstance(filename, Path):
        filename = Path(filename)
    return filename.suffix == extension


def read_tiff(filename):
    img = tifffile.imread(filename)
    #img = np.moveaxis(img, 0, -1)
    return img 


def read_nd2(filename, bundle_axes='zcyx'):
    with ND2Reader(filename) as images:
        if bundle_axes is not None:
            images.bundle_axes = bundle_axes
            images = images[0]
            return images
        return images

def read_img(filename, do_3D=False):
    if not isinstance(filename, (str, Path)):
        raise TypeError(f'filename should be of type str of pathlib.Path, but is {type(filename)}.')
    if isinstance(filename, str):
        filename = Path(filename)
    if filename.suffix == '.tif':
        return tifffile.imread(str(filename))
    elif filename.suffix == '.nd2':
        bundle_axes = 'zcyx' if do_3D else 'cyx'
        return read_nd2(str(filename), bundle_axes=bundle_axes)

def read_2mirror_img(filename):
    img = tifffile.imread(filename)
    _, Z, H, W = img.shape
    img = img.reshape(img, (2*Z, H, W))
    return img[0::2,], img[1::2,]

def get_files_in_folder(path, file_extension):
	if not isinstance(path, Path):
		path = Path(path)
	files = sorted([f for f in path.iterdir() if f.is_file() and \
        f.suffix == file_extension and not str(f.name).startswith('.')])
	return files


def get_subfolders(path):
	if not isinstance(path, Path):
		path = Path(path)
	folders = [f for f in path.iterdir() if f.is_dir()]
	return folders


def extract_cell_position_from_file_name(path):
        file_name = Path(path).stem
        position = [int(s) for s in re.split('X|Y|Z|-',file_name) if s.isdigit()]
        return position

# Image processing
def max_proj(img, axis=0, keepdims=False):
    img = img.max(axis=axis, keepdims=keepdims)
    if isinstance(img, torch.Tensor):
        img = img.values
    return img


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


def unsqueeze_to_ndim(img, n_dim):
    if len(img.shape) < n_dim:
        img = torch.unsqueeze(img,0) if isinstance(img, torch.Tensor) else np.expand_dims(img,0)
        unsqueeze_to_ndim(img, n_dim)
    return img


def squeeze_to_ndim(img, n_dim):
    if len(img.shape) > n_dim:
        img = torch.squeeze(img) if isinstance(img, torch.Tensor) else np.squeeze(img)
        unsqueeze_to_ndim(img, n_dim)
    return img


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


# Plotting
def show_dataset_examples(datasets, keys=['train', 'val', 'test'],n_examples = 5):
    '''Show a few examples from the train, val and test sets'''
    dataset_sizes = {key: len(datasets[key]) for key in keys}
    example_idx = {}
    plt.figure(figsize=(16,9))
    for k, key in enumerate(keys):
        example_idx[key] = random.sample(range(dataset_sizes[key]), k=n_examples)
        for i in range(n_examples):
            sample = datasets[key][example_idx[key][i]]
            plt.subplot(len(keys),n_examples, k*n_examples + i+1)
            plt.title(f"{key.capitalize()} sample | class {sample[1]}")
            plt.imshow(
                torch.mean(sample[0],0),
                cmap='gray',
                vmin=0,
                vmax=1
                )
    plt.show()


# Image transformations
def get_padding(image,shape=None):    
    _, h, w = image.shape
    if shape is None:
        shape = (np.max([h, w]),np.max([h, w]))
    h_padding = (shape[0] - h) / 2
    v_padding = (shape[1] - w) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(r_pad), int(t_pad), int(b_pad))
    return padding


class ShapePad(object):
    def __init__(self, shape=(128,128), fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.shape = shape
        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return torch.nn.functional.pad(img, get_padding(img,self.shape), mode=self.padding_mode,value=self.fill)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


class SquarePad(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return torch.nn.functional.pad(img, get_padding(img), mode=self.padding_mode,value=self.fill)
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class AugmentContrast(object):
    def __init__(self, contrast_range, channel_dim=0, preserve_range=True, per_channel=True, p_per_channel=0.5):
        self.contrast_range = contrast_range
        self.channel_dim = channel_dim
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
    
    def __call__(self, img):
        n_channels = img.shape[self.channel_dim]
        r1, r2 = self.contrast_range
        shape = torch.ones(img.ndim, dtype=int).tolist()
        if self.per_channel:
            shape[self.channel_dim] = n_channels
            factor = (r1 - r2) * torch.rand(shape) + r2
        else:
            factor = (r1 - r2) * torch.rand(shape) + r2
            shape[self.channel_dim] = n_channels
            factor = factor.repeat(shape)
        
        m = img.min()
        M = img.max()
        axis = list(range(img.ndim))
        axis.remove(self.channel_dim)
        augment_channel = torch.rand(shape) <= self.p_per_channel
        factor = torch.where(augment_channel, factor, torch.ones(shape))
        img = (img - img.mean(dim=axis, keepdim=True))*factor + img.mean(dim=axis, keepdim=True)
        if not self.preserve_range:
            m = img.min()
            M = img.max()
        img = img.clip(min=m, max=M)
        return img

class AugmentBrightness(object):
    def __init__(self, mu, sigma, channel_dim=0, preserve_range=True, per_channel=True, p_per_channel=0.5):
        self.mu = mu
        self.sigma = sigma
        self.channel_dim = channel_dim
        self.preserve_range = preserve_range
        self.per_channel = per_channel
        self.p_per_channel = p_per_channel
    
    def __call__(self, img):
        n_channels = img.shape[self.channel_dim]
        shape = torch.ones(img.ndim, dtype=int).tolist()
        if self.per_channel:
            shape[self.channel_dim] = n_channels
            rnd_nb = torch.randn(shape)*self.sigma + self.mu
        else:
            rnd_nb = torch.randn(shape)*self.sigma + self.mu
            shape[self.channel_dim] = n_channels
            rnd_nb = rnd_nb.repeat(shape)
        augment_channel = torch.rand(shape) <= self.p_per_channel
        m = img.min()
        M = img.max()
        img = img + augment_channel*rnd_nb
        if not self.preserve_range:
            m = img.min()
            M = img.max()
        img = img.clip(min=m, max=M)
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class ToTensorPerChannel(object):
    def __init__(self):
        pass

    def __call__(self, img):
        n_channels = img.shape[-1]
        out = torch.zeros(np.moveaxis(img, -1, 0).shape)
        for c in range(n_channels):
            out[c,...] = F.to_tensor(img[...,c]).div(img[...,c].max())
        return out

    def __repr__(self):
        return f"{self.__class__.__name__}()"

class NormalizeTensorPerChannel(object):
    def __init__(self, pmin, pmax, channel_dim=0, clip=True):
        self.pmin = pmin
        self.pmax = pmax
        self.channel_dim=channel_dim
        self.clip = clip
    
    def __call__(self, img):
        axis = list(range(img.ndim))
        axis.remove(self.channel_dim)
        pmin_values = torch.Tensor(np.percentile(img, self.pmin, axis=axis, keepdims=True).astype(np.float32))
        pmax_values = torch.Tensor(np.percentile(img, self.pmax, axis=axis, keepdims=True).astype(np.float32))
        img = (img - pmin_values)/(pmax_values-pmin_values)
        if self.clip:
            img = img.clip(min=0, max=1)
        return img
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"

class SelectChannels(object):
    def __init__(self, channels2use) -> None:
         self.channels2use = channels2use

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): image from which channels should be selected.
        Returns:
            img (torch.Tensors): image with only the requested channels.
        """
        n_channels = img.shape[0]
        if len(self.channels2use) > n_channels:
            raise ValueError(f'The number of requested channels (channels2use = {self.channels2use}) ' \
                f'exceeds the number of channels present in the image ({n_channels}).')
        return img[np.r_[self.channels2use],:,:]
    
    def __repr__(self):
        return f"{self.__class__.__name__}()"


# Datasets
def get_class_weights_counts(dataset):
    total = 0
    if isinstance(dataset, ImageFolder):
        classes = range(len(dataset.classes))
        imgs_class_pair_list = dataset.imgs
    elif isinstance(dataset, torch.utils.data.dataset.Subset):
        classes = range(len(dataset.dataset.classes))
        imgs_class_pair_list = [dataset.dataset.imgs[idx] for idx in dataset.indices]
    elif isinstance(dataset, DatasetFromDataFrame):
        classes = dataset.df['y'].unique()
        class_list = [dataset.df.iloc[idx]['y'] for idx in range(len(dataset))]
        counts = {c: 0 for c in classes}
        for c in class_list:
                total += 1
                counts[c] += 1
        ratios = [counts[c]/total for c in counts]
        weights = [1 - ratio for ratio in ratios]
        return ratios, weights, counts
    counts = {c: 0 for c in classes}
    for img_class_pair in imgs_class_pair_list:
            total += 1
            counts[img_class_pair[1]] += 1
    ratios = [counts[c]/total for c in counts]
    weights = [1 - ratio for ratio in ratios]
    return ratios, weights, counts


class ImageFolderWithPaths(ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __getitem__(self, index):
        return super(ImageFolderWithPaths, self).__getitem__(index) + (self.imgs[index][0],)


class CustomDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        data = self.dataset[index]
        img = data[0]
        if img.ndim > 3:
            img = squeeze_to_ndim(img, n_dim=3)
            img = np.moveaxis(img, 0, -1)
        if self.transform:
            x = self.transform(img.astype(np.float32))
        else:
            x = img
        y = data[1]

        if len(data) == 3:
            path = data[2]
            return x, y, path
        else:
            return x, y
    
    def __len__(self):
        return len(self.dataset)


def my_collate(batch):
    batch = list(filter(lambda x : not x[0].isnan().any(), batch))
    return default_collate(batch)

class DatasetFromDataFrame(Dataset):
    def __init__(self, df, loader, target_names=None, transform=None):
        self.df = df
        self.loader = loader
        self.transform = transform
        if target_names is None:
            self.target_names = sorted(df['target'].unique())
        else:
            self.target_names = target_names
        self.target_to_idx = {name: idx for (idx, name) in enumerate(self.target_names)}
        self.get_target_counts()
        self.get_target_ratios()
        self.get_target_weights()

    def __getitem__(self, index):
        data = self.df.iloc[index,:]

        ROI_img_path = data['ROI_img_path'],
        img = self.loader(ROI_img_path)

        target = self.target_to_idx[data['target']]

        metadata = {
            'dataset': data['dataset'],
            'ROI_img_path': data['ROI_img_path'],
            'ROI_img_name': data['ROI_img_name'],
            'row': data['row'],
            'col': str(data['col']),
            'pos': str(data['pos']),
            'target': data['target'],
            # 'density': data['density'],
            # 'ratio_astro': data['ratio_astro'],
            # 'ratio_SHSY5Y': data['ratio_SHSY5Y']
        }
        
        if img.ndim > 3:
            img = squeeze_to_ndim(img, n_dim=3)
            img = np.moveaxis(img, 0, -1)
        if self.transform:
            x = self.transform(img.astype(np.float32))
        else:
            x = img

        return x, target, metadata
    
    def __len__(self):
        return len(self.df.index)

    def get_target_counts(self):
        """
        Get the count of each target in the dataset

        :param df: dataset of samples with column 'y' representing the sample target label
        :type df: pandas.DataFrame
        :param targets: Possible target labels in the dataset
        :type targets: Union[list, 1D numpy.ndarray]
        :returns target_counts: count for each target label in the dataset
        :type target_counts: 1D numpy.ndarray
        """
        n_targets = len(self.target_names)
        target_counts = np.zeros((n_targets,))
        for i, target in enumerate(self.target_names):
            target_counts[i] = len(self.df[self.df['target'] == target])
        self.target_counts = target_counts
    
    def get_target_ratios(self):
        """
        Get the ratio of each target in the dataset. Assumes that target_counts.sum()
        returns the total number of samples in the dataset.

        :param target_counts: count for each target label in the dataset
        :type target_counts: 1D numpy.ndarray
        :returns target_ratios: ratio of each target label in the dataset
        :type target_ratios: 1D numpy.ndarray
        """
        total_count = self.target_counts.sum()
        target_ratios = np.array([x/total_count for x in self.target_counts])
        self.target_ratios = target_ratios
    
    def get_target_weights(self):
        self.target_weights = 1 - self.target_ratios
    
    def get_df_groups(self, grouping_vars):
        """
        Returns group for each sample in df based on the defined grouping variables
        :param df: Dataframe with sample info variables
        :type df: pandas.DataFrame
        :param grouping_vars: categorical variable name(s) in df (column(s)) by which 
            the data should be grouped
        :type grouping_vars: Union[str, list]
        :returns groups: group index for each sample in df
        :type groups: pandas.Series
        """
        if isinstance(grouping_vars, str):
            grouping_vars = [grouping_vars]
        for grouping_var in grouping_vars:
            if grouping_var not in self.df.columns:
                raise ValueError(f'The grouping variable "{grouping_var}" '
                    'is net present in the files_csv DataFrame')
        groups = self.df.groupby(grouping_vars).ngroup()
        return groups

    @staticmethod
    def get_grouped_train_test_split(X, y, groups, n_splits, shuffle, random_state):
        """
        Get stratified grouped train test split.
        :param X: data to split
        :type X: array-like of shape (n_samples, n_features), see sklearn.model_selection.StratifiedGroupKFold
        :param y: labels for the samples in X
        :type y: array-like of shape (n_samples,), see see sklearn.model_selection.StratifiedGroupKFold
        :param groups: group index of samples in X
        :type groups: array-like of shape (n_samples,), see see sklearn.model_selection.StratifiedGroupKFold
        :param shuffle: Whether to shuffle each class its samples before splitting into batches.
        :type shuffle: bool
        :random_state: When shuffle is True, random_state affects the ordering of the indices, 
            which controls the randomness of each fold for each class. Otherwise, leave random_state 
            as None. Pass an int for reproducible output across multiple function calls.
        :type random_state: int or RandomState instance
        """
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        splits = [x for x in cv.split(X, y, groups)]
        return splits

    @staticmethod
    def get_stratified_cost(target_ratios_total, target_ratios_subsets, weights=None):
        """
        Calculates a (weigted) cost based on the difference in the stratification between the dataset
        and multiple subsets of that dataset. A weight can be given for each dataset, resulting in a higher
        cost for a higher weight.

        :param target_ratios_total: target label ratios of the whole dataset
        :type target_ratios_total: 1D numpy.ndarray
        :param target_ratios_subsets: target label ratios for multiple subsets of the dataset.
        :type target_ratios_subsets: Union[list, tuple] containing 1D numpy.ndarray's of the target ratios of
            each subset.
        :returns cost: (weighted) cost for deviating from target_ratios_total in the subsets
        :type cost: float
        """
        if weights is None:
            n_subsets = len(target_ratios_subsets)
            weights = [1/n_subsets for _ in range(n_subsets)]
        cost = np.sum([
            w * np.linalg.norm(target_ratios_total-x) for (x, w) in zip(target_ratios_subsets, weights)
            ])
        return cost
    
    def get_stratified_grouped_split(self, desired_split, grouping_vars, random_state):
        """
        Get stratified grouped train, val, test split of a dataset in which it's attempted to split the
        dataset as closesly as possible to the desired dataset split, while balancing the
        following objectives:
        - Assigning groups defined by the grouping_vars uniquely to only one of the train, val or test subset.
            This objective is enforced.
        - Retaining the stratifications of the targets in whole dataset (df['target']) in the train, val and test 
            subset as closely as possible.
        
        :param df: dataset with 'target' label column to be split
        :type df: pandas.DataFrame
        :param desired_split: Desired proporation of data to be assigned to the train, test and val subset
        :type desired_split: 3-element list
        :param grouping_vars: categorical variable name(s) in df (column(s)) by which 
            the data should be grouped
        :type grouping_vars: Union[str, list]
        :random_state: When shuffle is True, random_state affects the ordering of the indices, 
            which controls the randomness of each fold for each class. Otherwise, leave random_state 
            as None. Pass an int for reproducible output across multiple function calls.
        :type random_state: int or RandomState instance
        """
        
        n_total = len(self)
        groups = self.get_df_groups(grouping_vars=grouping_vars)
        n_test_splits = np.round(1/desired_split[-1]).astype(int)
        test_ratio = 1/n_test_splits
        remaining = 1 - test_ratio
        n_train_val_splits = np.round(1/(1 - desired_split[0]/remaining))
        val_ratio_subset = 1/n_train_val_splits
        train_ratio_subset = 1 - val_ratio_subset
        print(
            f'Intended part of dataset assigned to train, val and test set: ',\
            f'{train_ratio_subset*remaining:.02f}, {val_ratio_subset*remaining:.02f}, {test_ratio:.02f}'
            )

        test_splits = self.get_grouped_train_test_split(
            self.df,
            self.df['target'],
            groups,
            n_splits=n_test_splits,
            shuffle=False,
            random_state=None
            )
        print(test_splits)
        splits = []
        for split in test_splits:
            train_val_idx, test_idx = split
            actual_test_ratio = len(test_idx)/n_total
            actual_remaining = 1 - actual_test_ratio
            actual_n_train_val_splits = int(np.round(actual_remaining/desired_split[1]))

            n_sub_splits = int(actual_n_train_val_splits)
            sub_splits = self.get_grouped_train_test_split(
                self.df.iloc[train_val_idx],
                self.df['target'].iloc[train_val_idx],
                groups.iloc[train_val_idx],
                n_splits=n_sub_splits,
                shuffle=False,
                random_state=None
                )
            costs = np.empty((len(sub_splits),))
            for i, sub_split in enumerate(sub_splits):
                train_idx, val_idx = sub_split
                train_idx, val_idx = train_val_idx[train_idx], train_val_idx[val_idx]
                dataset_train = DatasetFromDataFrame(self.df.iloc[train_idx], loader=self.loader)
                dataset_val = DatasetFromDataFrame(self.df.iloc[val_idx], loader=self.loader)
                costs[i] = self.get_stratified_cost(
                    self.target_ratios, 
                    (dataset_train.target_ratios, dataset_val.target_ratios),
                    weights=None
                    )
            best_idx = np.argmin(costs)
            train_idx, val_idx = sub_splits[best_idx]
            train_idx, val_idx = train_val_idx[train_idx], train_val_idx[val_idx]
            splits.append([train_idx, val_idx, test_idx])
        return splits

class PredictionDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.files = get_files_in_folder(self.root_dir, file_extension='.tif')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = str(self.files[idx])
        img = read_tiff(file_name)

        if self.transform:
            img = self.transform(img)

        return img, file_name