from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import random
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt

from nucleocentric.utils.utils import get_row_col_pos
from nucleocentric.utils.transforms import squeeze_to_ndim
from nucleocentric.utils.io import read_tiff, get_files_in_folder

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


def get_samples_df(input_path, layout_filepath, target_names=None, file_extension='.tif', 
                   GT_data_file=None, mixed_culture=False, density_range=False):
    """
    Returns a DataFrame with the dataset path, the ROI image path
    row, col, pos and label of each sample found in subfolders of the input_path
    The subfolder names represent the labels of the samples contained in it.

    Args:
        input_path (str or Path): Path to the root directory containing image subfolders
        layout_filepath (str): Path to Excel file containing well plate layout information
        file_extension (str, optional): File extension of images to load. Defaults to '.tif'
        GT_data_file (str, optional): Path to CSV file containing ground truth data. Required for mixed_culture=True. Defaults to None
        mixed_culture (bool, optional): Whether to process mixed culture data. Defaults to False
        density_range (bool, optional): Whether to include density range information. Defaults to False

    Returns:
        pandas.DataFrame: DataFrame containing:
            - dataset: Path to dataset root directory
            - ROI_img_path: Full path to each ROI image
            - ROI_img_name: Name of ROI image file
            - row: Well plate row
            - col: Well plate column  
            - pos: Position within well
            - target: Target label from layout file
            - true_condition: Ground truth label (only if mixed_culture=True)

    """
    input_path = Path(input_path) if not isinstance(input_path, Path) else input_path

    # Define well plate layout
    culture_layout = pd.read_excel(layout_filepath, sheet_name='Culture', index_col=0)

    img_folders = [folder.stem for folder in input_path.iterdir() \
        if folder.is_dir() and not folder.stem.startswith('.')]
    dataset, sample_paths = [], []
    rows, cols, positions, targets, densities, ratios_astro, ratios_SHSY5Y = [], [], [], [], [], [], []
    for img_folder in tqdm(img_folders):
        img_folder = input_path.joinpath(img_folder)
        img_folder_name = img_folder.name
        row, col, pos = get_row_col_pos(img_folder_name)
        target = culture_layout.loc[row, col]

        sample_paths_img_folder = sorted(img_folder.rglob('*' + file_extension))
        sample_paths_img_folder = [str(sample) for sample in sample_paths_img_folder if not sample.stem.startswith('.')]
        
        n_samples = len(sample_paths_img_folder)
        sample_paths.extend(sample_paths_img_folder)
        dataset.extend([str(input_path)]*n_samples)
        rows.extend([row]*n_samples)
        cols.extend([col]*n_samples)
        positions.extend([pos]*n_samples)
        targets.extend([target]*n_samples)
    
    # Build DataFrame
    sample_names = [Path(sample).stem for sample in sample_paths]
    df = pd.DataFrame.from_dict({
        'dataset': dataset,
        'ROI_img_path': sample_paths,
        'ROI_img_name': sample_names,
        'row': rows,
        'col': cols,
        'pos': positions,
        'target': targets,
        })
    
    # Filter by target names if provided
    if target_names is not None:
        df = df.loc[df['target'].isin(target_names)]

    if mixed_culture:
        assert GT_data_file is not None, "GT_data_file is required for mixed culture"
        df_GT = pd.read_csv(GT_data_file)
        df_GT = df_GT[['ROI_img_name','true_condition']] 
        df_GT.columns = ['ROI_img_name','true_condition']
        df_GT = df_GT.dropna(axis = 0)
        df = pd.merge(df, df_GT, on='ROI_img_name')  #merge the true condition with the feature dataframe
        df = df.loc[df['target'] == 'co-culture'] 
        print(df)
        df = df[df['true_condition'].str.contains('inconclusive')==False]
        df = df.drop(['target'], axis = 1)
        df.rename(columns = {'true_condition':'target'}, inplace = True)
    if density_range:
        assert GT_data_file is not None, "GT_data_file is required for density range"
        df_GT = pd.read_csv(GT_data_file)
        df_GT = df_GT[['ROI_img_name','true_density']]
        df = pd.merge(df, df_GT, on='ROI_img_name')  #merge the true condition with the feature dataframe
        true_density = df['true_density'].tolist()
        thr_95 = float(np.quantile(true_density,0.95)) 
        thr_80 = float(np.quantile(true_density,0.8)) 
        thr_60 = float(np.quantile(true_density,0.6))
        thr_40 = float(np.quantile(true_density,0.4))
        thr_20 = float(np.quantile(true_density,0.2))
        print(thr_20, thr_40, thr_60, thr_80, thr_95)
        true_density = [
            (df['true_density'] < thr_20),
            (df['true_density'] >= thr_20) & (df['true_density'] < thr_40),
            (df['true_density'] >= thr_40) & (df['true_density'] < thr_60),
            (df['true_density'] >= thr_60) & (df['true_density'] < thr_80),
            (df['true_density'] >= thr_80) & (df['true_density'] < thr_95),
            (df['true_density'] >= thr_95)
        ]
        density_cat = ['<20', '20-40', '40-60', '60-80','80-95', '>95']
        df['density_cat'] = np.select(true_density, density_cat)
        sample_density = min(df['density_cat'].value_counts())
        print(df['density_cat'].value_counts())
        df = df.groupby("density_cat").sample(n=int(sample_density), random_state=0)

    return df 


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
            'target': data['target']
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