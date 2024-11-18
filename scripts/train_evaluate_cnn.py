from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import torch
import warnings
import pandas as pd
import math
from torchvision.models import resnet50
from torchvision.transforms import Compose
from torch.utils.data import DataLoader

import sys
sys.path.append(str(Path(__file__).parent.parent))

from nucleocentric import (
    set_random_seeds,
    seed_worker,
    load_custom_config,
    read_tiff,
    get_samples_df,
    DatasetFromDataFrame,
    my_collate,
    ToTensorPerChannel,
    SquarePad,
    Resize,
    NormalizeTensorPerChannel,
    AugmentBrightness,
    AugmentContrast,
    SelectChannels,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomRotation,
    train_model,
    test_accuracy
)

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
    
        # Stratify dataset in train/validation/test 
        target_names = args.target_names
        df, = get_samples_df(
            args.input_path,
            layout_filepath=args.layout,
            target_names=target_names,
            GT_data_file=args.GT_data_file,
            mixed_culture=args.mixed_culture,
            density_range=args.density_range
            )
        total_count = len(df)
        print(f'Targets: {target_names}')
        print(f'Total number of samples: {total_count}')
        df['well'] = df['ROI_img_name'].str[0:-5]
        print(len(df['well'].unique()))

        n_test_wells = math.ceil(len(df['well'].unique())*0.3)
        if (n_test_wells < 2):
            n_test_wells = 2
        n_val_wells = math.ceil(len(df['well'].unique())*0.1)
        if (n_val_wells < 2):
            n_val_wells = 2
        n_train_wells = math.ceil(len(df['well'].unique())*0.6)
        if (n_train_wells < 2):
            n_train_wells = 2
        print(n_test_wells, n_train_wells, n_val_wells)
        test_wells = df.groupby("target").sample(n=int(n_test_wells/2), random_state=seed); test_wells = test_wells['well'].unique()
        test_df = df[df['well'].isin(test_wells)]; val_df = df[~df['well'].isin(test_wells)]
        val_wells = val_df.groupby("target").sample(n=int(n_val_wells/2), random_state=seed); val_wells = val_wells['well'].unique()
        val_df = df[df['well'].isin(val_wells)]
        train_df = df[~df['well'].isin(val_wells)]; train_df = train_df[~train_df['well'].isin(test_wells)]
        print(len(train_df), len(test_df), len(val_df))
        ## subsample the dataset, make sure both classes are both of equal size
        if (args.sample == 'max'):
            sample_train = min(train_df['target'].value_counts()); sample_val = min(val_df['target'].value_counts()); sample_test = min(test_df['target'].value_counts())
            test_df = test_df.groupby("target").sample(n=int(sample_test), random_state=seed); train_df = train_df.groupby("target").sample(n=int(sample_train), random_state=seed); val_df = val_df.groupby("target").sample(n=int(sample_val), random_state=seed)            
        elif (args.sample == 'oversampling'):
            sample_train = max(train_df['target'].value_counts()); sample_val = max(val_df['target'].value_counts()); sample_test = max(test_df['target'].value_counts())
            test_df = test_df.groupby("target").sample(n=int(sample_test), random_state=seed, replace = True); train_df = train_df.groupby("target").sample(5000, random_state=seed, replace = True); val_df = val_df.groupby("target").sample(n=int(sample_val), random_state=seed, replace = True)            
        elif(args.sample == 'unequal'):
            test_df = test_df; train_df = train_df; val_df = val_df
        else:
            sample_train = int(args.sample); sample_test = int(args.sample)/3; sample_val = int(args.sample)/10
            test_df = test_df.groupby("target").sample(n=int(sample_test), random_state=seed); train_df = train_df.groupby("target").sample(n=int(sample_train), random_state=seed); val_df = val_df.groupby("target").sample(n=int(sample_val), random_state=seed)
        test_df['subset'] = 'test'; train_df['subset'] = 'train';val_df['subset'] = 'val'
        df = train_df.append(pd.DataFrame(data = val_df), ignore_index=True); df = df.append(pd.DataFrame(data = test_df), ignore_index=True)
        df = df.reset_index(drop = True)            
        print(len(test_df), len(train_df), len(val_df))
        train_df['well'].to_csv('train_instances.csv')


        for cv_idx in range(3): ## number of cross validaiton iterations

            datasets = {
                'train': DatasetFromDataFrame(train_df, loader=read_tiff, transform=transforms['train']),
                'val': DatasetFromDataFrame(val_df, loader=read_tiff, transform=transforms['val']),
                'test': DatasetFromDataFrame(test_df, loader=read_tiff, transform=transforms['test'])
            }

            for key in datasets:
                print(f'{key.capitalize()} set target to index: {datasets[key].target_to_idx}')
                print()

            dataset_split_info = {}
            for key in ('train', 'val', 'test'):
                count = len(datasets[key])
                dataset_split_info[key]={
                    'count': count,
                    'portion': count/total_count,   
                    'target_counts': datasets[key].target_counts.tolist(),
                    'target_ratios': datasets[key].target_ratios.tolist(),
                    }
            print('Dataset split info:')
            print(dataset_split_info)

            # Create dataloaders
            batch_size = args.batch_size
            n_workers = 0
            dataloaders={
                'train': DataLoader(
                    datasets['train'],
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=n_workers,
                    generator=generator,
                    worker_init_fn=seed_worker,
                    collate_fn=my_collate
                    ),
                'val': DataLoader(
                    datasets['val'],
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=n_workers,
                    generator=generator,
                    worker_init_fn=seed_worker,
                    collate_fn=my_collate
                    ),
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
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print("The model will be running on", device, "device")
            loss_fn_train = torch.nn.CrossEntropyLoss(weight=torch.Tensor(datasets['train'].target_weights).to(device))
            loss_fn_eval = torch.nn.CrossEntropyLoss(weight=torch.Tensor(datasets['val'].target_weights).to(device))

            # Train model
            exp_id = f'{exp_nr:02d}_{cv_idx:02d}'
            train_results = train_model(
                model, dataloaders, loss_fn_train, loss_fn_eval, optimizer, scheduler,
                device, args.n_epochs, batch_size, exp_id
                )

            # Test model on test set
            print('Testing trained model on test set')
            model.load_state_dict(train_results['best_val_model_state_dict'])
            accuracy, _, cf_matrix, test_df = test_accuracy(
                model,
                dataloaders['test'],
                device=device
            )
            test_df = pd.concat(
                (test_df, pd.Series(['test']*len(test_df), name='dataset_split')),
                axis=1
                )
            tnr, fpr, fnr, tpr = 100*cf_matrix.ravel()

            cf_matrix = pd.DataFrame(cf_matrix, index = [i for i in test_df.target_names],
                        columns = [i for i in test_df.target_names])

            print(f'Test set accuracy {accuracy}')
            print()
            print('Confusion matrix:')
            print(cf_matrix)
            print()

            # Save results in .pth and .json file
            if args.save:
                results_pth = {
                    'random_seed': seed,
                    'input_path': str(args.input_path),
                    'target_names': args.target_names,
                    'split': args.split,
                    'dataset_split_info': dataset_split_info,
                    'image_size': args.image_size,
                    'channels2use': args.channels2use,
                    'transforms': transforms,
                    'device': device,
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'epochs': args.n_epochs,
                    'loss_fn_train': loss_fn_train,
                    'loss_fn_eval': loss_fn_eval,
                    'test_accuracy': accuracy,
                    'tnr': tnr,
                    'fpr': fpr,
                    'fnr': fnr,
                    'tpr': tpr,
                    'train_loss_each_epoch': train_results['train_loss_each_epoch'],
                    'val_loss_each_epoch': train_results['val_loss_each_epoch'],
                    'best_val_loss': train_results['best_val_loss'],
                    'best_val_accuracy': train_results['best_val_accuracy'],
                    'best_val_epoch': train_results['best_val_epoch'],
                    'final_val_accuracy': train_results['final_val_accuracy'],
                    'final_model_state_dict': train_results['final_model_state_dict'],
                    'final_optimizer_state_dict': train_results['final_optimizer_state_dict'],
                    'final_scheduler_state_dict': train_results['final_scheduler_state_dict'],
                    'best_val_model_state_dict': train_results['best_val_model_state_dict'],
                    'best_val_optimizer_state_dict': train_results['best_val_optimizer_state_dict'],
                    'best_val_scheduler_state_dict': train_results['best_val_scheduler_state_dict']
                }
                results_json = {
                    'random_seed': seed,
                    'input_path': str(args.input_path),
                    'target_names': args.target_names,
                    'split': args.split,
                    'dataset_split_info': dataset_split_info,
                    'image_size': args.image_size,
                    'channels2use': args.channels2use,
                    'transforms': repr(transforms),
                    'device': repr(device),
                    'batch_size': args.batch_size,
                    'learning_rate': args.learning_rate,
                    'epochs': args.n_epochs,
                    'loss_fn_train': repr(loss_fn_train),
                    'loss_fn_eval': repr(loss_fn_eval),
                    'test_accuracy': accuracy,
                    'tnr': tnr,
                    'fpr': fpr,
                    'fnr': fnr,
                    'tpr': tpr,
                    'train_loss_each_epoch': train_results['train_loss_each_epoch'],
                    'val_loss_each_epoch': train_results['val_loss_each_epoch'],
                    'best_val_loss': train_results['best_val_loss'],
                    'best_val_accuracy': train_results['best_val_accuracy'],
                    'best_val_epoch': train_results['best_val_epoch'],
                    'final_val_accuracy': train_results['final_val_accuracy']
                }
                # Load data info inf present
                try:
                    data_info_file_name = str(args.input_path.joinpath('..','info.json'))
                    with open(data_info_file_name) as json_file:
                        data_info = json.load(json_file)
                    results_pth['data_info'] = data_info
                    results_json['data_info'] = data_info
                except:
                    warnings.warn(
                        f"No info.json file found in the input path {Path(custom_config['data_path']).joinpath(args.input_path)}. " +
                        "No info on the dataset is added to the output .pth and .json file."
                        )
                # Store .pth file with results
                output_path = Path(args.output_path)
                file_name = output_path.joinpath(time_stamp + f'_model_experiment_{exp_id}')
                torch.save(results_pth, str(file_name) + '.pth')
                # Store .json file with results
                with open(str(file_name) + '.json', 'w', encoding='utf-8') as f:
                    json.dump(results_json, f, ensure_ascii=False, indent=4)
                # Store .csv file with test results per test sample
                train_val_test_df = pd.concat(
                    (train_results['best_train_df'], train_results['best_val_df'], test_df),
                    axis=0,
                    ignore_index=True
                    )
                train_val_test_df.to_csv(str(file_name) + '.csv', index=False)

            # Track result of random seed
            experiments[exp_nr][cv_idx]={
                'random_seed': seed,
                'test_accuracy': accuracy,
                'tnr': tnr,
                'fpr': fpr,
                'fnr': fnr,
                'tpr': tpr,
                }
    
    print('Experiments overview')
    print('---------------------------------------')

    for exp_nr in experiments:
        test_accuracies = []
        tnrs = []
        tprs = []
        for cv_idx in experiments[exp_nr]:
            accuracy = experiments[exp_nr][cv_idx]['test_accuracy']
            tnr = experiments[exp_nr][cv_idx]['tnr']
            tpr = experiments[exp_nr][cv_idx]['tpr']
            test_accuracies.append(accuracy)
            tnrs.append(tnr)
            tprs.append(tpr)
            print(f"Experiment {exp_nr} | CV index {cv_idx} | test accuracy: {accuracy}%", file=open('output.txt', 'a'))
            print(f"Experiment {exp_nr} | CV index {cv_idx} | true negative rate: {tnr}%", file=open('output.txt', 'a'))
            print(f"Experiment {exp_nr} | CV index {cv_idx} | true positive rate: {tpr}%", file=open('output.txt', 'a'))

        print('---------------------------------------')

        test_accuracies = np.array(test_accuracies)
        tnrs = np.array(tnrs)
        tprs = np.array(tprs)
        print(f"Experiment {exp_nr} | Test accuracy mean: {test_accuracies.mean():.2f}% +/- {test_accuracies.std():.2f}%", file=open('output.txt', 'a'))
        print(f"Experiment {exp_nr} | True negative rate: {tnrs.mean():.2f}% +/- {tnrs.std():.2f}%", file=open('output.txt', 'a'))
        print(f"Experiment {exp_nr} | True positive rate: {tprs.mean():.2f}% +/- {tprs.std():.2f}%", file=open('output.txt', 'a'))
        print('---------------------------------------')
        print()


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True, 
        help='Relative path from the data_path specified in the custom_config.json file. \
            The input_path holds the class subfolders.')
    parser.add_argument('--output_path', type=str, required=True,
        help='Output path where results are stored')
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
    parser.add_argument('--save', action='store_true',
        help='Save results and model after training')
    parser.add_argument('--mixed_culture', action='store_true',
        help='mixed coculture')
    parser.add_argument('--density_range', action='store_true',
        help='stratify per density')
    parser.add_argument('-b', '--batch_size', type=int, default=256,
        help='Number of samples in each batch during training')
    parser.add_argument('-l', '--learning_rate', type=float, default=0.0001,
        help='Learning rate for model training')
    parser.add_argument('--sample', type=str, required=True,
        help='Sample number')

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