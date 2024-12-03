# Unbiased identification of cell identity in dense mixed neural cultures
This repository contains the scripts used in De Beuckeleer et al. 2024 (eLife). In this work, we have applied the high-content image-based morphological cell profiling technique (Cell Painting) to heterogenous mixed cultures. We performed morphological profiling at the single-cell level by segmenting individual cells and associating Cell Painting image data to the ground-truth phenotype using cyclic fluorescent staining. Single cell types were then classified using shallow machine learning and more advanced deep learning. 
Link to ELife paper: https://elifesciences.org/reviewed-preprints/95273v1#s4

Execution of the data analysis should be performed by running the scripts (located in the `scripts` folder) in a sequential order. Each script contains several argparse arguments. These arguments are parameters that can be tuned at several points in the data-analysis pipeline to result in an optimal cell classification performance. The action of each argument is listed below for each script. A short description of the argument's action can also be found at the bottom of the code of each script. The flowchart details the order in which the scripts should be executed based on the type of analysis you want to perform. 

![GitHub_flowchart](https://github.com/user-attachments/assets/c353d29c-d859-4457-b9c7-c30287e9474e)


An accompanying test dataset is included and can be downloaded using the following link: https://figshare.com/articles/dataset/Nucleocentric-Profiling/27141441?file=49522557. The zip-file contains 2x 480 raw image files of cell painting and associated ground-truth images of a co-culture of 1321N1 and SH-SY5Y cells. The folder also contains a layout-file detailing the plate layout. A step-by-step tutorial on the test-dataset can be found below. 

# Dependencies
This project is implemented in Python. It is highly recommended create a virtual environment in which dependencies can be installed. We provide an `environment.yml` file from which a virtual environment with all dependencies can be installed using `conda`. We used Python 3.8, since we found it the easiest for creating a virtual environment that includes compatible versions of PyTorch, Tensorflow, Keras, Stardist, Cellpose and Napari (for visualization).

Below we provide the terminal/command prompt commands for setting up the virtual environment using `conda` and the provided `environment.yml` file. `conda`'s default solver might struggle to find out which packages need to be installed. In case installation takes too long using `conda`, we recommend using [Miniforge](https://github.com/conda-forge/miniforge#miniforge3). When using Miniforge, run the commands form the Miniforge Prompt and replace `conda` by `mamba` in the commands below.

Create a virtual environment from the `environment.yml` file:
```
conda env create -f environment.yml
```

Activate the created environment:
```
conda activate nucleocentric-py3.8
```


Example of environment installation: 
![miniforge_install](https://github.com/user-attachments/assets/f44c5a0e-0f2d-4152-9904-5302433668ce)


To support cross-platform compatibility, we did not add GPU-based dependencies for Nvidia GPUs. By making use of CUDA, the speed of model training and inference can be significantly increased. If you have an NVIDIA GPU, follow the commands below to add GPU support:

Uninstall Pytorch:
```
conda uninstall pytorch
```

Re-install Pytorch and torchvision, and install pytorch-cuda:
```
conda install -c pytorch -c nvidia pytorch=1.13.0 torchvision=0.14.0 pytorch-cuda=11.6
```


# Available scripts and arguments

Each script executes distinct steps explained below. The scripts make use of the source code provided in the `nucleocentric` subfolder. Each script should be executed from the terminal/command prompt in the root directory of this repository. Make sure the virtual environment is activated (see instructions above).

## 1) Set filenames

All filenames should be in the format "X-YY-ZZ" where X refers to the well plate row, Y refers to the well plate column and Z refers to the position of the image in the well.
For the images in the test dataset, the script included can be used to rename all files correctly. 

```
python scripts\renamer.py \                                    # Script based on Cellpose for whole cell segmentation                    
    --filepath [INPUT_PATH] \                                  # Path to the folder with images to rename                                # Required
    --file_extension_before .nd2 \                             # Extention type of the input images                                      # Default '.nd2'
    --file_extension_after .nd2 \                              # Extention type of the output images                                     # Default '.nd2'
```


## 2) Segmentation
**Full cell segmentation**

Scripts for cell and nuclei segmentation. Identification of ROIs in the CP image is the first required step for further single cell classification. Either cell segmentation (for whole cell crops) or nuclear segmentation (for nuclear and nucleocentric crops) is performed. Cell segmentation makes use of Cellpose, while nuclei segmentation relies on Stardist. We found the latter to be most reliable in dense cultures.

```
python scripts\segment_cells.py \                                    # Script based on Cellpose for whole cell segmentation                    
    -i [INPUT_PATH] \                                                # Path to the raw input images                                            # Required
    -o [OUTPUT_PATH] \                                               # Path where the output masks and resulting images can be stored          # Required
    --file_extension [FILE_EXTENSION] \                              # Extention type of the input images                                      # Default '.tif'
    --save_masks \                                                   # Store the masks after segmentation as a separate .tif file              # Not required
    --pmin 1 --pmax 99 \                                             # Top and bottom percentile for input image normalization                 # default is 1 (bottom) and 99 (top)
    --gpu \                                                          # Try to use the GPU for segmentation if this is available                # Not required
    --net_avg \                                                      # Use the average prediction of 4 models to improve segmentation          # Not required
    --channels2use 0 1 2 3 \                                         # Channel numbers of the input image to use for segmentation              # Default '0 1 2 3 4 5'
    --target cyto \                                                  # What region will be segmented (cyto or nuclei)                          # Default 'cyto'
    --diameter 100                                                   # Estimated diameter of the cells in the image                            # Not required
```

**Nuclei segmentation** 
```
python scripts\segment_nuclei.py \                                   # Script based on Stardist for nuclei segmentation
    -i [INPUT_PATH] \                                                # Path to the raw input images                                            # Required
    -o [OUTPUT_PATH] \                                               # Path where the output masks and resulting images can be stored          # Required
    --file_extension [FILE_EXTENSION] \                              # Extention type of the input images                                      # Default '.nd2'
    --pmin 1 --pmax 99 \                                             # Top and bottom percentile for input image normalization                 # default is 1 (bottom) and 99 (top)
    --gpu \                                                          # Try to use the GPU for segmentation if this is available                # Not required
    --probability 0.6 \                                              # Probability threshold for ROI detection                                 # Default '0.6'
    --nms_thresh 0.3                                                 # NMS threshold for handling overlapping objects                          # Default '0.3'
```


## 2) Preprocessing
Operations performed on the whole image. 

**GT alignment**

GT2CP alignment: necessary for mixed cultures where post-hoc staining was performed to identify the true cell phenotype. The GT image is translated in X and Y to overlap at single-cell level with the CP image and masks identified in step 1.
```
python scripts\align_GT2CP.py \                                        # Script for the alignment of CP images with post-hoc ICC images.
    --CP_path [PATH_TO_CP_IMAGES] \                                    # Path to the raw input images                                            # Required
    --GT_path [PATH_TO_GT_IMAGS] \                                     # Path to the raw ground-truth images                                     # Required
    --GT_name [NAME_OF_STAINING] \                                     # Name of the staining, used as folder name                               # Required
    --masks_path [PATH_TO_MASKS] \                                     # Path to the segmentation masks                                          # Required
    --channels2use_CP [CP_CHANNEL_IDX_TO_BE_USED_FOR_ALIGNMENT] \      # Channels used of the CP image for the alignment                         # Default '0 1 2 3 4 5'
    --channels2use_GT [GT_CHANNEL_IDX_TO_BE_USED_FOR_ALIGNMENT] \      # Channels used of the GT image for the alignment                         # Default '0 1 2 3 4 5'
    --file_extension_imgs [FILE_EXTENSION_CP_AND_GT_IMAGES] \          # File extension of the raw CP and GT images                              # Default '.nd2'
    --file_extension_masks [FILE_EXTENSION_MASKS] \                    # File extension of the segmentation masks                                # Default '.tif'
    --output_path [OUTPUT_PATH]                                        # Path where the folder containing aligned images will be stored          # Required
```

**Cropping of ROIs** 

Cropping of ROIs: individual cells (ROIs) are cropped out of the original image as patches based on segmentation masks. Patch size (in pixel) for full cell = 192, for nucleus and nucleocentric crops: 60. These crops are given as an input to the CNN for cell classification.
```
python scripts\crop_ROIs.py \                                          # Script cropping smaller patches out of the original image to give as an input to the CNN to train
    --CP_path [PATH_TO_CP_IMAGES] \                                    # Path to the aligned CP input images                                     # Required
    --masks_path [PATH_TO_MASKS] \                                     # Path to the aligned segmentation masks                                  # Required
    --file_extension_imgs [FILE_EXTENSION_CP_AND_GT_IMAGES] \          # File extension of the aligned CP images                                 # Default '.tif'
    --file_extension_masks [FILE_EXTENSION_MASKS] \                    # File extension of the aligned segmentation masks                        # Default '.tif'
    --output_path [OUTPUT_PATH] \                                      # Path where the crops will be stored                                     # Required
    --masked_patch \                                                   # Set area outside the segmentation mask to zero                          # Used for 'whole cell' and 'nuclear' regions
    --patch_size 192                                                   # Size of the crop (box around the centroid of the segmentation mask)     # Default '192' (for whole cell), should be set to '60' for nuclear and nucleocentric crops
```

**Handcrafted feature extraction**

Get intensity and texture features: handcrafted features extracted for RF and UMAP. The intensity features (if extracted from ground-truth images) are used to threshold and determine the true phenotype of a cell. The intensity and texture features (if extracted from CP images) are used as input for the random forest.

```
python scripts\get_intensity_features.py \                             # Script extracting handcrafted intensity and shape features out of the input images. Used for GT thresholding (on GT images) and input for random forest (on CP images)
    --GT_path [PATH_TO_IMAGES] \                                       # Path to the aligned input images (can be GT or CP images)               # Required
    --GT_channel_names [NAMES_OF_CHANNELS] \                           # Names of the channels in the input image (in order)                     # Required
    --masks_path [PATH_TO_MASKS] \                                     # Path to the aligned segmentation masks                                  # Required
    --layout [PATH_TO_LAYOUT_FILE] \                                   # Path to the layout file                                                 # Required
    --file_extension_GT [FILE_EXTENSION_IMAGES] \                      # File extension of the aligned input images                              # Default '.tif'
    --file_extension_masks [FILE_EXTENSION_MASKS] \                    # File extension of the aligned segmentation masks                        # Default '.tif'
    --output_path [OUTPUT_PATH] \                                      # Path where the output .csv file containing the features will be stored  # Required
    --masked_patch \                                                   # Set area outside the segmentation mask to zero                          # Used for 'whole cell' and 'nuclear' regions
    --patch_size 192                                                   # Size of the box around the centroid of the segmentation mask            # Default '192' (for whole cell), should be set to '60' for nuclear and nucleocentric crops
```

```
python scripts\get_texture_features.py \                               # Script extracting texture features out of the input CP images. Used as input for the random forest.
    --CP_path [PATH_TO_IMAGES] \                                       # Path to the aligned CP images                                            # Required
    --masks_path [PATH_TO_MASKS] \                                     # Path to the aligned segmentation masks                                   # Required
    --file_extension_imgs [FILE_EXTENSION_CP_IMAGES] \                 # File extension of the aligned CP images                                  # Default '.tif'
    --file_extension_masks [FILE_EXTENSION_MASKS] \                    # File extension of the aligned segmentation masks                         # Default '.tif'
    --output_path [OUTPUT_PATH] \                                      # Path where the output .csv file containing the features will be stored   # Required
    --masked_patch \                                                   # Set area outside the segmentation mask to zero                           # Used for 'whole cell' and 'nuclear' regions
    --patch_size 192                                                   # Size of the box around the centroid of the segmentation mask             # Default '192' (for whole cell), should be set to '60' for nuclear and nucleocentric crops
```

## 3) Classification

```
python script\train_evaluate_CNN.py \                                 # Script for training and evaluating a CNN on the CP crops
    --input_path [PATH_TO_CP_CROPS] \                                 # Path to the CP crops, corresponds to the output path of the crop_ROIs.py script.                                                        # Required
    --output_path [OUTPUT_PATH] \                                     # Path where the trained model, .json and predictions will be stored          # Required
    --target_names [CULTURE_NAMES_TO_TRAIN_ON] \                      # Names of the classes that need to be predicted (as in layout file)          # Required
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \                 # Layout file (excel), reflecting the contents of the well plate              # Required
    --GT_data_file [PATH_TO_GT_FILE] \                                # .csv file containing the true phenotype of each ROI                         # Required
    --epochs [NUMBER_OF_TRAINING_EPOCHS | 50] \                       # Number of epochs (iterations) for model training                            # Default '50'
    --image_size [IMAGES_ARE_RESIZED_TO_THIS_SIZE | 128] \            # Size (in pixels) to which the input crops are resized for training          # Default '128'                
    --channels2use [CHANNELS_USED_IN_TRAINING] \                      # Fluorescent channels used for model training                                # Default '0 1 2 3 4 5'
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY] \                 # Seed for data sampling and splitting in train/test/validation sets          # Default '0'              
    --save \                                                          # Indicate to save the .pth file (model)                                      # Not required
    --mixed_culture                                                   # Indicate if you are predicting from a co-culture                            # Not required
    --batch_size [NUMBER_OF_SAMPLES_TO_PROCESS_PER_ITERATION | 100] \ # Number of samples to process per iteration                                  # Default '100'
    --learning_rate [LEARNING_RATE | 0.0001]                          # Learning rate of the CNN                                                    # Default '0.0001'
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS]                           # How many ROIs of each target class are included in the training set         # Required
```

```
python scripts\train_evaluate_RF.py \                                 # Script for training and evaluating a random forest on a feature dataframe
    --data_file [PATH_TO_FEATURE_DATA] \                              # Path to file containing ROI features (output from get_intensity_features.py and get_texture_features.py)        # Required
    --regions2use [REGIONS_AS_INPUT] \                                # Which regions to use for training and evaluation (cell, cyto, nucleus or all)                                   # Required
    --channels2use [CHANNELS_AS_INPUT] \                              # Which channels to use for training and evaluation (DAPI, FITC, Cy3, Cy5 or all)                                 # Required
    --mixed_culture                                                   # Indicate if you are predicting from a co-culture                                                                # Not required
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS] \                         # How many ROIs of each target class are included in the training set                                             # Required
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY]                   # Seed for data sampling and splitting in train/test/validation sets                                              # Default '0'
```

## 4) Evaluation
```
python scripts\embeddings.py \                                        # Script for extracting the feature embeddings from a trained CNN model
    --input_path [PATH_TO_CP_CROPS] \                                 # Input path to the CP crops you want to extract embeddings from                # Required
    --output_path [OUTPUT_PATH] \                                     # Output path where you want to store the .csv file with embeddings             # Required
    --model_file [PATH_TO_MODEL_FILE] \                               # Path to the trained CNN network (.pth file)                                   # Required
    --target_names [CULTURE_NAMES_TO_TRAIN_ON] \                      # Names of the prediction classes (as in the layout file)                       # Required
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \                 # Path to the layout file (excel), reflecting the contents of the plate layout  # Required
    --GT_data_file [PATH_TO_GT_FILE] \                                # .csv file containing the true phenotype of each ROI                           # Required
    --image_size [IMAGES_ARE_RESIZED_TO_THIS_SIZE | 128] \            # Size (in pixels) to which the input crops are resized for training            # Default '128' 
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY] \                 # Seed for data sampling and splitting in train/test/validation sets            # Default '0'  
    --mixed_culture                                                   # Indicate if you are predicting from a co-culture                              # Not required
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS]                           # How many ROIs of each target class are included in the training set           # Required
```
```
python scripts\grad_cam.py \                                          # Script for generating GradCAM heatmaps of individual crops
    --input_path [PATH_TO_CP_CROPS] \                                 # Input path to the CP crops you want to extract embeddings from                # Required
    --model_file [PATH_TO_MODEL_FILE] \                               # Path to the trained CNN network (.pth file)                                   # Required
    --target_names [CULTURE_NAMES_TO_TRAIN_ON] \                      # Names of the prediction classes (as in the layout file)                       # Required
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \                 # Path to the layout file (excel), reflecting the contents of the plate layout  # Required
    --GT_data_file [PATH_TO_GT_FILE] \                                # .csv file containing the true phenotype of each ROI                           # Required
    --image_size [IMAGES_ARE_RESIZED_TO_THIS_SIZE | 128] \            # Size (in pixels) to which the input crops are resized for training            # Default '128' 
    --channels2use [CHANNELS_USED] \                                  # Fluorescent channels used for model training                                  # Default '0 1 2 3 4 5'
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY] \                 # Seed for data sampling and splitting in train/test/validation sets            # Default '0'
    --mixed                                                           # Indicate if you are predicting from a co-culture                              # Not required
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS]                           # How many ROIs of each target class are included in the training set           # Required
```
# Step-by-step tutorial

## (1) Get started by installing the environment and downloading the demo dataset and scripts
Install the environment.yml file as described above. Make sure the environment is activated when running the scripts.

The demo dataset (zip-file, 20 GB) contains 480 cell painting images and 480 matched ground truth images. It is associated with a layout file describing the plate layout.
Download, unzip and store the data locally (e.g., C:\Users\...\ELIFE2024\CP). The Cell Painting images have 4 channels. The ground truth images have 2 channels (BrdU and EdU).

Download the scripts and store them locally (e.g., in the same folder as the images). We recommend using Visual Studio Code (VSCode) to run the scripts. Upon starting VSCode, activate the correct folder by selecting File > Open folder > [PATH_TO_SCRIPTS\...\PARENT]. Make sure to activate the parent folder containing the 2 subfolders 'nucleocentric' and 'scripts' (both found in this GitHub). Within the 'scripts' folder, all scripts can be found as described within this README and in the flowchart above. Run the code in the terminal by selecting Terminal > New Terminal and typing the commands below.

Example of file structure:

![file_location](https://github.com/user-attachments/assets/963d9e5f-f30c-48ea-abbe-d0e4c1d3f568)

In this example, the data and scripts are stored locally. The images (CP and GT) are stored in the respective folders. Within the folder 'Nucleocentric-Profiling', the subfolders 'nucleocentric' and 'scripts' are located. This parent folder (Nucleocentric-Profiling) is opened in VSCode. The layout file is also found together with the data and scripts.

## (2) Rename files
For both the folder containing CP and GT images, rename files to the X-YY-ZZ format. 
Run the code in the cmd terminal:
```
python scripts\renamer.py --filepath C:\Users\...\ELIFE2024\CP
```

## (3) Segment individual cells
From the CP images, individual ROIs are segmented. This can be either from the full cell or only the nucleus. 
Run the code in the cmd terminal:
* Cell segmentation:
```
python scripts\segment_cells.py -i C:\Users\...\ELIFE2024\CP -o C:\Users\...\ELIFE2024\CP\cell_masks --file_extension .nd2 --save_masks --gpu --net_avg --channels2use 0 1 2 3 --target cyto
```
* Nucleus segmentation:
```
python scripts\segment_nuclei.py -i C:\Users\...\ELIFE2024\CP -o C:\Users\...\ELIFE2024\CP\nuclei_masks --file_extension .nd2 --gpu
```
This segmentation step generates as output the segmentation masks.

## (4) Ground truth alignment
The ground truth information was acquired by cyclic staining. This means the multiwell plate is imaged twice. In between imaging cycles, the plate is taken off the microscope stage. As a result, small translational shifts can occur. We need to overlay information from all imaging rounds. Therefore, we perform this alignment step where the ground truth images are translated to perfectly match the CP images and masks.
Run the code in the cmd terminal: 
```
python scripts\align_GT2CP.py --CP_path C:\Users\...\ELIFE2024\CP --GT_path C:\Users\...\ELIFE2024\GT --GT_name EdU_BrdU --masks_path C:\Users\...\ELIFE2024\CP\cell_masks --channels2use_CP 0 --channels2use_GT 0 --file_extension_imgs .nd2  --file_extension_masks .tif --output_path C:\Users\...\ELIFE2024
```
This step generates a folder containing aligned CP images, aligned GT images and aligned masks.

## (5) Get ground truth datafile
Intensity information can be extracted from the GT images. Based on this intensity data and the presence and absence of fluorescent signal for either marker, a threshold is defined that determines which class the ROIs belong to. 
The intensity information is extracted by running 'get_intensity_features.py' in the cmd terminal given the GT images as input:
```
python scripts\get_intensity_features.py --GT_path C:\Users\...\ELIFE2024\[timestamp]\GT --GT_channel_names EdU BrdU --masks_path C:\Users\...\ELIFE2024\[timestamp]\masks --layout C:\Users\...\ELIFE2024\layout.xlsx --file_extension_GT .tif --file_extension_masks .tif --output_path C:\Users\...\ELIFE2024\ --masked_patch --patch_size 60
```
This step generates a .csv file containing features (columns) for each ROI (rows). This file is loaded into the 'set_GT_threshold.py' script. This script cannot be run in the terminal since it requires manual examination of the thresholds.

Example of ground-truth identification by thresholding (with intensity features extracted using 'get_intensity_features.py'): 

![image](https://github.com/user-attachments/assets/be65a0e6-8fe3-435e-ab98-f748bf611c41)

As an output of this step, a 'GT_data.csv' file is generated. This file contains for each ROI, a unique ROI identifier associated with a 'true condition'. This true condition refers to the true cell class or is 'undefined' when no unequivocal ground truth staining is present. 

## (6) Crop ROIs
The individual ROIs are cropped out of the cell painting image. These crops can be given to the CNN as input. A crop is defined by 2 parameters: patch size and masking. The patch size refers to how large the crop is (expressed in pixels surrounding the centroid of the segmentation mask). Masking determines whether the background (all pixels outside of the segmentation mask) are put to zero or not. Run this code in the cmd terminal.
* Full cell crops:
```
python scripts\crop_ROIs.py --CP_path C:\Users\...\ELIFE2024\[timestamp]\CP --masks_path C:\Users\...\ELIFE2024\[timestamp]\masks --file_extension_imgs .tif --file_extension_masks .tif --output_path C:\Users\...\ELIFE2024\[timestamp]\crops --masked_patch --patch_size 192
```
* Nuclear crops:
```
python scripts\crop_ROIs.py --CP_path C:\Users\...\ELIFE2024\[timestamp]\CP --masks_path C:\Users\...\ELIFE2024\[timestamp]\masks --file_extension_imgs .tif --file_extension_masks .tif --output_path C:\Users\...\ELIFE2024\[timestamp]\crops --masked_patch --patch_size 60
```
* Nucleocentric crops:
```
python scripts\crop_ROIs.py --CP_path C:\Users\...\ELIFE2024\[timestamp]\CP --masks_path C:\Users\...\ELIFE2024\[timestamp]\masks --file_extension_imgs .tif --file_extension_masks .tif --output_path C:\Users\...\ELIFE2024\[timestamp]\crops --patch_size 60
```
This step results in small ROI crops.

## (7) Features for random forest
Handcrafted features can be extracted from the original image. These features need to be extracted from the CP images. There are 2 separate scripts that can be merged using the unique identifier. Run this code in the cmd terminal.
* Intensity and shape features:
```
python scripts\get_intensity_features.py --GT_path C:\Users\...\ELIFE2024\[timestamp]\CP --GT_channel_names DAPI FITC Cy3 Cy5 --masks_path C:\Users\...\ELIFE2024\[timestamp]\masks --layout C:\Users\...\ELIFE2024\layout.xlsx --file_extension_GT .tif --file_extension_masks .tif --output_path C:\Users\...\ELIFE2024 --masked_patch --patch_size 192
```
* Texture features:
```
python scripts\get_texture_features.py --CP_path C:\Users\...\ELIFE2024\[timestamp]\CP --masks_path C:\Users\...\ELIFE2024\[timestamp]\masks --file_extension_imgs .tif --file_extension_masks .tif --output_path C:\Users\...\ELIFE2024 --masked_patch --patch_size 192
```
The resulting .csv file can be given as input to the random forest or can be used to create PCA, UMAP, ...

## (8) Random Forest
This code builds a random forest based on the hancrafted features. Run this code in the cmd terminal.
```
python scripts\train_evaluate_RF.py --data_file C:\Users\...\ELIFE2024\features.csv --regions2use all --channels2use all --mixed_culture --sample 2000 --random_seed 0 --GT_data_file C:\Users\...\ELIFE2024\GT_data.csv
```

## (9) Convolutional neural network
This code trains a CNN using the image crops as input. Run this code in the cmd terminal.
```
python scripts\train_evaluate_cnn.py --input_path C:\Users\...\ELIFE2024\[timestamp]\crops --output_path C:\Users\...\ELIFE2024\predictions --target_names astro SHSY5Y --layout C:\Users\...\ELIFE2024\layout.xlsx --GT_data_file C:\Users\...\ELIFE2024\GT_data.csv --channels2use 0 1 2 3 --random_seed 0 --save --mixed_culture --batch_size 100 --sample 2000
```
This script outputs a .csv file containining the prediction results, the trained model and a json file with metadata.

## (10) Understanding the CNN
To understand how the CNN makes it prediction, it is possible to extract the feature embeddings of the model and plot the embeddings using UMAP. Or gradCAM maps can be used to visualize where the attention of the CNN goes to.
* feature embedding extraction:
```
python scripts\embeddings.py --input_path C:\Users\...\ELIFE2024\[timestamp]\crops --output_path C:\Users\...\ELIFE2024\predictions --model_file C:\Users\...\ELIFE2024\predictions\....pth --target_names astro SHSY5Y --layout C:\Users\...\ELIFE2024\layout.xlsx --GT_data_file C:\Users\...\ELIFE2024\GT_data.csv --random_seed 0 --mixed_culture --sample 2000
```
This script results in a .csv file containing the feature embeddings, unique ROI identifier and true class condition.
* gradCAM heatmap:
```
python scripts\grad_cam.py --input_path C:\Users\...\ELIFE2024\[timestamp]\crops --model_file C:\Users\...\ELIFE2024\predictions\....pth --target_names astro SHSY5Y --layout C:\Users\...\ELIFE2024\layout.xlsx --GT_data_file C:\Users\...\ELIFE2024\GT_data.csv --channels2use 0 1 2 3 --random_seed 0 --mixed_culture --sample 
```

## Validation
The installation, test dataset and code have been validated on 3 separate devices
* HP Pavilion: Microsoft Windows 10.0.19045 Build 19045. Intel(R) Core(TM) i7-8550U CPU @ 1.80GHz, 2001 Mhz, 4 Core(s), 8 Logical Processor(s). GPU NVIDIA GeForce GTX 1050.
* MS-7D04: Microsoft Windows 11 Enterprise 10.022631 Build 22631. 11th Gen Intel(R) Core(TM) i9-11900K @ 3.50GHz, 3504 Mhz, 8 Core(s), 16 Logical Processor(s). GPU NVIDIA GeForce RTX 3090.
* Apple MacBook Pro (2021) with 32GB RAM. macOS Sonoma Version 14.5. Apple M1 Pro chip.

 
## HOW TO CITE
If you use this repository, please cite the paper:
```
@article{Beuckeleer_2024,
    author={Beuckeleer, Sarah De and De Looverbosch, Tim Van and Den Daele, Johanna Van and Ponsaerts, Peter and De Vos, Winnok H.}, 
    title={Unbiased identification of cell identity in dense mixed neural cultures}, 
    url={http://dx.doi.org/10.7554/eLife.95273.1},
    DOI={10.7554/elife.95273.1}, 
    publisher={eLife Sciences Publications, Ltd}, 
    year={2024}, 
    month=mar
}
```
