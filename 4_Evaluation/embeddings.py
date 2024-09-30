import json
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import re
import warnings
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18, resnet50
from torchvision.transforms import (Compose, ToTensor, Resize,
                                    RandomRotation, RandomHorizontalFlip, RandomVerticalFlip)
from tqdm import tqdm
import math
import random
from sklearn.metrics import confusion_matrix
import pandas as pd
from util import get_files_in_folder, read_img, squeeze_to_ndim,create_composite2D

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


from util import (
    DatasetFromDataFrame, SquarePad, ToTensorPerChannel, NormalizeTensorPerChannel,
    SelectChannels, AugmentBrightness, AugmentContrast, load_custom_config, read_tiff, 
    seed_worker, set_random_seeds, my_collate
    )


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

def get_samples_df(input_path, layout_filepath, target_names, file_extension='.tif'):
    """
    Returns a DataFrame with the dataset path, the ROI image path
    row, col, pos and label of each sample found in subfolders of the input_path
    The subfolder names represent the labels of the samples contained in it.

    :param input_path: 
    :type input_path: Union[string, pathlib.Path]
    :returns df: DataFrame with dataset path, original image path, original image filename,
        sample image path, sample label, original image row, original image col, original image pos.
    :type df: pandas.DataFrame
    """
    input_path = Path(input_path) if not isinstance(input_path, Path) else input_path

    # Define well plate layout
    culture_layout = pd.read_excel(layout_filepath, sheet_name='Culture', index_col=0)
    # ratio_layout = pd.read_excel(layout_filepath, sheet_name='Ratio', index_col=0)
    # density_layout = pd.read_excel(layout_filepath, sheet_name='Density', index_col=0)

    img_folders = [folder.stem for folder in input_path.iterdir() \
        if folder.is_dir() and not folder.stem.startswith('.')]
    dataset, sample_paths = [], []
    rows, cols, positions, targets, densities, ratios_astro, ratios_SHSY5Y = [], [], [], [], [], [], []
    for img_folder in tqdm(img_folders):
        img_folder = input_path.joinpath(img_folder)
        img_folder_name = img_folder.name
        row, col, pos = get_row_col_pos(img_folder_name)
        target = culture_layout.loc[row, col]
        # Continue if not the right target
        # if target not in target_names:
        #     continue
        sample_paths_img_folder = sorted(img_folder.rglob('*' + file_extension))
        sample_paths_img_folder = [str(sample) for sample in sample_paths_img_folder if not sample.stem.startswith('.')]
        
        n_samples = len(sample_paths_img_folder)
        sample_paths.extend(sample_paths_img_folder)
        dataset.extend([str(input_path)]*n_samples)
        rows.extend([row]*n_samples)
        cols.extend([col]*n_samples)
        positions.extend([pos]*n_samples)
        targets.extend([target]*n_samples)
        # densities.extend([density_layout.loc[row, col]]*n_samples)
        # ratio_astro = ratio_layout.loc[row, col]
        # ratios_astro.extend([ratio_astro]*n_samples)
        # ratios_SHSY5Y.extend([100 - ratio_astro]*n_samples)
    
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
        # 'density': densities,
        # 'ratio_astro': ratios_astro,
        # 'ratio_SHSY5Y': ratios_SHSY5Y
        })
    df_test = df
    df_train = df
    if args.mixed:
        df_GT = pd.read_csv(args.GT_data_file)
        df_GT = df_GT[['ROI_img_name','true_condition']] 

        print(df_GT['ROI_img_name'])
        df_merged = pd.merge(df, df_GT, on='ROI_img_name')  #merge the true condition with the feature dataframe
        df = df_merged.copy()
        df = df_merged.loc[df_merged['target'] == 'co-culture'] 
        df = df[df['true_condition'].str.contains('inconclusive')==False]
        print(df) 
        df = df.drop(['target'], axis = 1)
        df.rename(columns = {'true_condition':'target'}, inplace = True)
        print(df)
        df_train = df
        df_test = df
    print(df.columns, df_test.columns, df_train.columns)
    return df, df_test, df_train


def test_accuracy(model, dataloader, device, loss_fn=torch.nn.CrossEntropyLoss()):
    model.eval()
    accuracy = 0.0
    loss = 0.0
    total = 0.0

    y_pred = []
    y_true = []
    
    with torch.no_grad():
        test_results = pd.DataFrame()
        for (images, labels, metadata) in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            # run the model on the test set to predict labels
            outputs = model(images)
            outputs = torch.nn.functional.softmax(outputs, dim=1)
            loss += loss_fn(outputs, labels)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

            y_pred.extend(predicted.data.cpu().numpy())
            y_true.extend(labels.data.cpu().numpy())

            data = pd.concat(
                (
                    pd.DataFrame.from_dict(metadata), 
                    pd.DataFrame.from_dict({
                        'predicted': predicted.data.cpu().numpy(),
                        'ground_truth': labels.data.cpu().numpy()
                        })
                ),
                axis=1
            )
            test_results = pd.concat((test_results, data), axis=0, ignore_index=True)




    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    loss = loss.item() / total

    # confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred, normalize='true')

    return accuracy, loss, cf_matrix, test_results



def main(args):
    # Time stap the training run
    time_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # Define the input and output path
    custom_config = load_custom_config()
    args.input_path = Path(custom_config['data_path']).joinpath(args.input_path)

    # Train model for each random seed
    experiments={}
    for exp_nr, seed in enumerate(args.random_seed):
        experiments[exp_nr] = {}
        print(f'Experiment {exp_nr} | Random seed: {seed}')
        
        # Set random seeds
        generator = set_random_seeds(seed=seed)

        # Define transformations for preprocessing and data augmentation
        image_size = (args.image_size, args.image_size)
        print(f'Training will be done using channels {args.channels2use}')
        transforms = {
        'train':Compose([
            ToTensorPerChannel(),
            SquarePad(),
            Resize(image_size),
            NormalizeTensorPerChannel(pmin=0.1, pmax=99.9),
            AugmentBrightness(
                mu=0.0, sigma=0.1, channel_dim=0,
                preserve_range=True, per_channel=True, p_per_channel=0.5
                ),
            AugmentContrast(
                contrast_range=(.75, 1.25),  channel_dim=0, 
                preserve_range=True, per_channel=True, p_per_channel=0.5
                ),
            NormalizeTensorPerChannel(pmin=0, pmax=100),
            RandomHorizontalFlip(),
            RandomVerticalFlip(),
            RandomRotation((-180,180)),
            SelectChannels(args.channels2use)
            ]),
        'val': Compose([
            ToTensorPerChannel(),
            SquarePad(),
            Resize(image_size),
            NormalizeTensorPerChannel(pmin=0.1, pmax=99.9),
            SelectChannels(args.channels2use)
            ]),
        'test': Compose([
            ToTensorPerChannel(),
            SquarePad(),
            Resize(image_size),
            NormalizeTensorPerChannel(pmin=0.1, pmax=99.9),
            SelectChannels(args.channels2use)
            ])
        }
        
        target_names = args.target_names
        df, _, _ = get_samples_df(args.input_path, target_names=target_names, layout_filepath=args.layout)
        dataset = DatasetFromDataFrame(df, loader=read_tiff)
        df = df.loc[df['target'] != 'co-culture'] 
        print(df)
        total_count = len(df)
        print(f'Targets: {dataset.target_names}')
        print(f'Total number of samples: {total_count}')
        test_df = df.copy()
        if args.sample != 'max':
            sample_test = int(args.sample)
        else: 
            sample_test = min(test_df['target'].value_counts())
        test_df = test_df.groupby("target").sample(n=int(sample_test), random_state=seed)
        test_df['subset'] = 'test'
        ROIS = test_df['ROI_img_name'].to_numpy()
        print(test_df)


        for cv_idx in range(1):
            datasets = {
                'test': DatasetFromDataFrame(test_df, loader=read_tiff, transform=transforms['test'])
            }

            for key in datasets:
                print(f'{key.capitalize()} set target to index: {datasets[key].target_to_idx}')
                print()

            # Create dataloaders
            batch_size = args.batch_size
            n_workers = 0
            dataloaders={
                'test': DataLoader(
                    datasets['test'],
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=n_workers,
                    generator=generator,
                    worker_init_fn=seed_worker,
                    collate_fn=my_collate
                    ),
            }
        
            # Create model, optimizer and lr scheduler
            n_targets = len(target_names)
            n_channels = len(args.channels2use)
            model = resnet50(pretrained=False)
            model.conv1 = torch.nn.Conv2d(
                n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
                )
            model.fc = torch.nn.Linear(2048, n_targets)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

            # Define execution device
            device = torch.device("cpu")
            print("The model will be running on", device, "device")
            model_file = torch.load(args.model_file)
            model.load_state_dict(model_file['final_model_state_dict'])
            target_layers = torch.nn.Sequential(*list(model.children())[:-1])
            target_layers.cuda()
            all_labels = []
            all_embeddings = []
            torch.cuda.empty_cache()
            model.eval()
            for data,labels,_ in tqdm(dataloaders['test']):
                new_labels = labels.numpy().tolist()
                all_labels += new_labels
                data = data.cuda()
                embeddings = target_layers(data.cuda())
                all_embeddings.append(np.reshape(embeddings.detach().cpu().numpy(),(len(new_labels),-1)))
            all_embeddings = np.vstack(all_embeddings)
            df = pd.DataFrame(all_embeddings)
            # df['ROI_img_name'] = ROIS
            # df['labels'] = all_labels
            filename = str(Path(args.output_path).joinpath('Embeddings.csv'))
            df.to_csv(filename, index=False)
            # np.save(output_path,all_embeddings)
            # np.save(output_path,np.array(all_labels))



def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True, 
        help='Relative path from the data_path specified in the custom_config.json file. \
            The input_path holds the class subfolders.')
    parser.add_argument('-o', '--output_path', type=str, required=True, 
        help='Relative path from the data_path specified in the custom_config.json file. \
            The input_path holds the class subfolders.')
    parser.add_argument('--model_file', type=str, required=True,
        help='Filepath of trained model state_dict')
    parser.add_argument('-t', '--target_names', type=str, nargs='+', required=True,
        help='Targets to use for training in the classification problem. Samples with other targets with be ignored')
    parser.add_argument('--layout', type=str, required=True,
        help='Path of the excel file describing the well plate layout')
    parser.add_argument('--GT_data_file', type=str, required=True,
        help='file containing ground truth')
    parser.add_argument('-e', '--n_epochs', type=int, default=50,
        help='Number of epochs during model training')
    parser.add_argument('-s', '--image_size', type=int, default=128,
        help='Size (H,W) of the input images for the model')
    parser.add_argument('--channels2use', nargs='+', type=int, choices=[0,1,2,3,4,5], default=[0,1,2,3,4,5],
		help='Specify the channels to use for training the classifier')    
    parser.add_argument('-r', '--random_seed', type=int, nargs='+', default=0,
        help='Random seeds to be used set for reproducibility. If multiple seeds are given, \
            multiple models are trained, each with its own random seed')
    parser.add_argument('--split', type=float, nargs=3, default=[0.60,0.10,0.30],
        metavar=('train', 'val', 'test'), help='Train, val and test split')
    parser.add_argument('--mixed', action='store_true',
        help='mixed coculture')
    parser.add_argument('-b', '--batch_size', type=int, default=256,
        help='Number of samples in each batch during training')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0001,
        help='Learning rate for model training')
    parser.add_argument('--sample', type=str, required=True,
        help='sample dataset')


    args = parser.parse_args()

    if isinstance(args.random_seed, int):
        # Turn into list
        args.random_seed = [args.random_seed]

    if isinstance(args.target_names, str):
        # Turn into list
        args.target_names = [args.target_names]

    if sum(args.split) != 1.0:
        raise ValueError(f'Train, val and test split should sum up to 1.0, ' \
            f'but got {args.split[0]} + {args.split[0]} + {args.split[0]} = {sum(args.split)}')

    args.channels2use = list(args.channels2use) if isinstance(args.channels2use, int) else args.channels2use

    return args

if __name__ == '__main__':
    args = parse_arguments()
    main(args)