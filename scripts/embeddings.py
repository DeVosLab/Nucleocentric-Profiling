from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.transforms import (Compose, Resize)
from tqdm import tqdm
import pandas as pd

import sys
sys.path.append(str(Path(__file__).parent.parent))

from nucleocentric import (
    load_custom_config,
    read_tiff,
    get_samples_df,
    DatasetFromDataFrame,
    my_collate,
    ToTensorPerChannel,
    SquarePad,
    Resize,
    NormalizeTensorPerChannel,
    SelectChannels
)


def main(args):
    # Define the input and output path
    custom_config = load_custom_config()
    args.input_path = Path(custom_config['data_path']).joinpath(args.input_path)

    # Train model for each random seed
    experiments={}
    for exp_nr, seed in enumerate(args.random_seed):
        experiments[exp_nr] = {}
        print(f'Experiment {exp_nr} | Random seed: {seed}')

        # Define transformations for preprocessing and data augmentation
        image_size = (args.image_size, args.image_size)
        print(f'Training will be done using channels {args.channels2use}')
        transforms =  Compose([
            ToTensorPerChannel(),
            SquarePad(),
            Resize(image_size),
            NormalizeTensorPerChannel(pmin=0.1, pmax=99.9),
            SelectChannels(args.channels2use)
        ])
        
        
        df, _, _ = get_samples_df(
            args.input_path,
            target_names=args.target_names,
            layout_filepath=args.layout,
            GT_data_file=args.GT_data_file,
            mixed_culture=args.mixed_culture
            )
        df = df.loc[df['target'] != 'co-culture'] 
        total_count = len(df)
        targets = args.target_names
        print(f'Targets: {targets}')
        print(f'Total number of samples: {total_count}')
        if args.sample != 'max':
            sample_test = int(args.sample)
        else: 
            sample_test = min(df['target'].value_counts())
        df = df.groupby("target").sample(n=int(sample_test), random_state=seed)
        print(f'Number of samples per target: {sample_test}')

        dataset = DatasetFromDataFrame(df, loader=read_tiff, transform=transforms)

        # Create dataloaders
        batch_size = args.batch_size
        n_workers = 0
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=n_workers,
            collate_fn=my_collate
        )
    
        # Create model
        n_targets = len(targets)
        n_channels = len(args.channels2use)
        model = resnet50(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
            n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
        model.fc = torch.nn.Linear(2048, n_targets)

        # Load trained weights
        model_file = torch.load(args.model_file)
        model.load_state_dict(model_file['final_model_state_dict'])

        # Get target layers to extract embeddings from
        model = torch.nn.Sequential(*list(model.children())[:-1])

        # Define execution device
        device = torch.device("cpu")
        model.to(device)
        print("The model will be running on", device, "device")
        
        # Extract embeddings
        all_labels = []
        all_embeddings = []
        model.eval()
        for data, labels, _ in tqdm(dataloader):
            new_labels = labels.numpy().tolist()
            all_labels += new_labels
            data = data.to(device)
            embeddings = model(data)
            all_embeddings.append(np.reshape(embeddings.detach().cpu().numpy(),(len(new_labels),-1)))
        all_embeddings = np.vstack(all_embeddings)
        df = pd.DataFrame(all_embeddings)
        filename = str(Path(args.output_path).joinpath('Embeddings.csv'))
        df.to_csv(filename, index=False)


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
        help='Targets labels to use. Samples with other targets with be ignored')
    parser.add_argument('--layout', type=str, required=True,
        help='Path of the excel file describing the well plate layout')
    parser.add_argument('--GT_data_file', type=str, required=True,
        help='file containing ground truth')
    parser.add_argument('-s', '--image_size', type=int, default=128,
        help='Size (H,W) of the input images for the model')
    parser.add_argument('--channels2use', nargs='+', type=int, choices=[0,1,2,3,4,5], default=[0,1,2,3,4,5],
		help='Specify the channels to use for training the classifier')
    parser.add_argument('--mixed_culture', action='store_true',
        help='mixed coculture')
    parser.add_argument('-b', '--batch_size', type=int, default=256,
        help='Number of samples in each batch during training')
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