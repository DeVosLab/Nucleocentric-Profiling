from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from torchvision.transforms import (Compose, Resize)

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


from nucleocentric.data.datasets import (
    get_samples_df, DatasetFromDataFrame, my_collate
)
from nucleocentric.utils.utils import (
    SquarePad, ToTensorPerChannel, NormalizeTensorPerChannel, SelectChannels,
    create_composite2D
)
from nucleocentric.utils.io import load_custom_config, read_tiff


def plot_results(image, gradcam):
    # Input image
    image_np = image.cpu().detach().numpy()
    img_composite = create_composite2D(image_np, channel_dim=0)
    fig, ax = plt.subplots(4, 2)
    ax[0,0].imshow(image[0,:,:], cmap='gray', interpolation=None) # interpolation='none'
    ax[0,0].axis('off'); ax[0, 0].set_title('Channel DAPI')
    ax[1,0].imshow(image[1,:,:], cmap='gray', interpolation=None) # interpolation='none'
    ax[1,0].axis('off'); ax[1, 0].set_title('Channel 488')
    ax[2,0].imshow(image[2,:,:], cmap='gray', interpolation=None) # interpolation='none'
    ax[2,0].axis('off'); ax[2, 0].set_title('Channel 555')
    ax[3,0].imshow(image[3,:,:], cmap='gray', interpolation=None) # interpolation='none'
    ax[3,0].axis('off'); ax[3, 0].set_title('Channel 638')
    ax[0,0].imshow(image[0,:,:], cmap='gray', interpolation=None) # interpolation='none'
    ax[0,0].axis('off'); ax[0, 0].set_title('Channel DAPI')

    ax[0,1].imshow(np.squeeze(img_composite))
    ax[0,1].axis('off'); ax[0, 1].set_title('Composite')

    # Output (overlay of masks on top of input)
    ax[1,1].imshow(gradcam, cmap='turbo', interpolation=None) # interpolation='none'
    ax[1,1].axis('off'); ax[1, 1].set_title('Gradcam')

    # And boxplot of the diameters of the segmented labels
    image = torch.mean(image,0)
    ax[2,1].imshow(image, cmap='gray', alpha = 1, interpolation=None) # I would add interpolation='none'
    ax[2,1].imshow(gradcam, cmap='turbo', alpha=0.4, interpolation=None) # interpolation='none'
    ax[2,1].axis('off'); ax[2, 1].set_title('Overlay')
    ax[3,1].axis('off')
    # Save figure
    plt.show()



def main(args):
        # Define the input and output path
    custom_config = load_custom_config()
    args.input_path = Path(custom_config['data_path']).joinpath(args.input_path)

    # Define transformations for preprocessing
    image_size = (args.image_size, args.image_size)
    print(f'Training will be done using channels {args.channels2use}')
    transforms = Compose([
        ToTensorPerChannel(),
        SquarePad(),
        Resize(image_size),
        NormalizeTensorPerChannel(pmin=0.1, pmax=99.9),
        SelectChannels(args.channels2use)
    ])

    # Get dataframe
    df = get_samples_df(
        args.input_path,
        target_names=args.target_names,
        layout_filepath=args.layout,
        GT_data_file=args.GT_data_file,
        mixed_culture=args.mixed_culture
        )
    total_count = len(df)
    targets = targets = args.target_names
    print(f'Targets: {targets}')
    print(f'Total number of samples: {total_count}')

    if args.sample != 'max':
        sample_test = int(args.sample)
    else: 
        sample_test = min(df['target'].value_counts())
    df = df.groupby("target").sample(n=int(sample_test), random_state=args.random_seed)
    print(f'Number of samples per target: {sample_test}')

    # Create dataset
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

    # Load pretrained weights
    model_file = torch.load(args.model_file)
    model.load_state_dict(model_file['final_model_state_dict'])

    # Define execution device
    device = torch.device("cpu")
    print("The model will be running on", device, "device")
    
    target_layers = torch.nn.Sequential(*list(model.children())[:-1])
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(0)]

    for (images, labels, metadata) in dataloader:
        images = images.to(device)
        image_name = metadata['ROI_img_path']
        print(image_name)
        input_tensor = images
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        img_plot = images[0, :, :, :] # change 
        plot_results(img_plot, grayscale_cam)                


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-i', '--input_path', type=str, required=True, 
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
    parser.add_argument('--mixed', action='store_true',
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