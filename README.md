# Unbiased identification of cell identity in dense mixed neural cultures
This repository contains the scripts used in De Beuckeleer et al. 2024 (eLife). In this work, we have applied the high-content image-based morphological cell profiling technique (Cell Painting) to heterogenous mixed cultures. We performed morphological profiling at the single-cell level by segmenting individual cells and associating Cell Painting image data to the ground-truth phenotype using cyclic fluorescent staining. Single cell types were then classified using shallow machine learning and more advanced deep learning. 
Link to ELife paper: https://elifesciences.org/reviewed-preprints/95273v1#s4

Execution of the data analysis should be performed by running the scripts in a sequential order. Each script contains several argparse arguments. These arguments are parameters that can be tuned at several points in the data-analysis pipeline to result in an optimal cell classification performance. The action of each argument is listed below for each script. A short description of the argument's action can also be found at the bottom of the code of each script. The flowchart details the order in which the scripts should be executed based on the type of analysis you want to perform. 

![GitHub_flowchart](https://github.com/user-attachments/assets/28b160a0-3fb4-4994-9f5b-1b4b95459a91)


An accompanying test dataset is included and can be downloaded using the following link: https://figshare.com/articles/dataset/Nucleocentric-Profiling/27141441?file=49522557. The zip-file contains 2x 480 raw image files of cell painting and associated ground-truth images of a co-culture of 1321N1 and SH-SY5Y cells. The folder also contains a layout-file detailing the plate layout. A step-by-step tutorial on the test-dataset can be found below. 

# Available scripts and arguments

Each folder within the repository groups the source code related to distinct steps explained below. The util file contains functions that can be called upon.

## 1) Segmentation
This folder contains scripts for cell and nuclei segmentation. Identification of ROIs in the CP image is the first required step for further single cell classification. Either cell segmentation (for whole cell crops) or nuclear segmentation (for nuclear and nucleocentric crops) is performed.

**Full cell segmentation**
```
python segment_cells.py \                                            # Script based on Cellpose for whole cell segmentation                    
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
python segment_nuclei.py \                                           # Script based on Stardist for nuclei segmentation
    -i [INPUT_PATH] \                                                # Path to the raw input images                                            # Required
    -o [OUTPUT_PATH] \                                               # Path where the output masks and resulting images can be stored          # Required
    --file_extension [FILE_EXTENSION] \                              # Extention type of the input images                                      # Default '.nd2'
    --pmin 1 --pmax 99 \                                             # Top and bottom percentile for input image normalization                 # default is 1 (bottom) and 99 (top)
    --gpu \                                                          # Try to use the GPU for segmentation if this is available                # Not required
    --probability 0.6 \                                              # Probability threshold for ROI detection                                 # Default '0.6'
    --overlap 0.03                                                   # Overlap threshold allowed for object detection                          # Default '0.03'

```


## 2) Processing
Operations performed on the whole image. 
GT2CP alignment: necessary for mixed cultures where post-hoc staining was performed to identify the true cell phenotype. The GT image is translated in X and Y to overlap at single-cell level with the CP image and masks identified in step 1.
Cropping of ROIs: individual cells (ROIs) are cropped out of the original image. Patch size (in pixel) for full cell = 192, for nucleus and nucleocentric crops: 60. Masked patch = True for full cell and nucleus. These crops are given as an input to the CNN for cell classification.
Get intensity and texture features: handcrafted features extracted for RF and UMAP. The intensity features (if extracted from ground-truth images) are used to threshold and determine the true phenotype of a cell. The intensity and texture features (if extracted from CP images) are used as input for the random forest.

**GT alignment** 
```
python align_GT2CP.py \                                                # Script for the alignment of CP images with post-hoc ICC images.
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
```
python crop_ROIs.py \                                                  # Script cropping smaller patches out of the original image to give as an input to the CNN to train
    --CP_path [PATH_TO_CP_IMAGES] \                                    # Path to the aligned CP input images                                     # Required
    --masks_path [PATH_TO_MASKS] \                                     # Path to the aligned segmentation masks                                  # Required
    --file_extension_imgs [FILE_EXTENSION_CP_AND_GT_IMAGES] \          # File extension of the aligned CP images                                 # Default '.tif'
    --file_extension_masks [FILE_EXTENSION_MASKS] \                    # File extension of the aligned segmentation masks                        # Default '.tif'
    --output_path [OUTPUT_PATH] \                                      # Path where the crops will be stored                                     # Required
    --masked_patch \                                                   # Set area outside the segmentation mask to zero                          # Used for 'whole cell' and 'nuclear' regions
    --patch_size 192                                                   # Size of the crop (box around the centroid of the segmentation mask)     # Default '192' (for whole cell), should be set to '60' for nuclear and nucleocentric crops
```
**Handcrafted feature extraction** 
```
python get_intensity_features.py \                                     # Script extracting handcrafted intensity and shape features out of the input images. Used for GT thresholding (on GT images) and input for random forest (on CP images)
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
python get_texture_features.py \                                       # Script extracting texture features out of the input CP images. Used as input for the random forest.
    --CP_path [PATH_TO_IMAGES] \                                       # Path to the aligned CP images                                            # Required
    --masks_path [PATH_TO_MASKS] \                                     # Path to the aligned segmentation masks                                   # Required
    --file_extension_imgs [FILE_EXTENSION_CP_IMAGES] \                 # File extension of the aligned CP images                                  # Default '.tif'
    --file_extension_masks [FILE_EXTENSION_MASKS] \                    # File extension of the aligned segmentation masks                         # Default '.tif'
    --output_path [OUTPUT_PATH] \                                      # Path where the output .csv file containing the features will be stored   # Required
    --masked_patch \                                                   # Set area outside the segmentation mask to zero                           # Used for 'whole cell' and 'nuclear' regions
    --patch_size 192                                                   # Size of the box around the centroid of the segmentation mask             # Default '192' (for whole cell), should be set to '60' for nuclear and nucleocentric crops
```

Example of ground-truth identification by thresholding (with intensity features extracted using 'get_intensity_features.py')
![image](https://github.com/user-attachments/assets/be65a0e6-8fe3-435e-ab98-f748bf611c41)


## 3) Classification

```
python train_evaluate_CNN.py \                                        # Script for training and evaluating a CNN on the CP crops
    --input_path [PATH_TO_CP_CROPS] \                                 # Path to the CP crops                                                        # Required
    --output_path [OUTPUT_PATH] \                                     # Path where the trained model, .json and predictions will be stored          # Required
    --target_names [CULTURE_NAMES_TO_TRAIN_ON] \                      # Names of the classes that need to be predicted (as in layout file)          # Required
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \                 # Layout file (excel), reflecting the contents of the well plate              # Required
    --GT_data_file [PATH_TO_GT_FILE] \                                # .csv file containing the true phenotype of each ROI                         # Required
    --epochs [NUMBER_OF_TRAINING_EPOCHS | 50] \                       # Number of epochs (iterations) for model training                            # Default '50'
    --image_size [IMAGES_ARE_RESIZED_TO_THIS_SIZE | 128] \            # Size (in pixels) to which the input crops are resized for training          # Default '128'                
    --channels2use [CHANNELS_USED_IN_TRAINING] \                      # Fluorescent channels used for model training                                # Default '0 1 2 3 4 5'
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY] \                 # Seed for data sampling and splitting in train/test/validation sets          # Default '0'              
    --save \                                                          # Indicate to save the .pth file (model)                                      # Not required
    --mixed                                                           # Indicate if you are predicting from a co-culture                            # Not required
    --batch_size [NUMBER_OF_SAMPLES_TO_PROCESS_PER_ITERATION | 100] \ # Number of samples to process per iteration                                  # Default '100'
    --learning_rate [LEARNING_RATE | 0.0001]                          # Learning rate of the CNN                                                    # Default '0.0001'
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS]                           # How many ROIs of each target class are included in the training set         # Required
```

```
python train_evaluate_RF.py \                                         # Script for training and evaluating a random forest on a feature dataframe
    --data_file [PATH_TO_FEATURE_DATA] \                              # Path to file containing ROI features (output from get_intensity_features.py and get_texture_features.py)        # Required
    --regions2use [REGIONS_AS_INPUT] \                                # Which regions to use for training and evaluation (cell, cyto, nucleus or all)                                   # Required
    --channels2use [CHANNELS_AS_INPUT] \                              # Which channels to use for training and evaluation (DAPI, FITC, Cy3, Cy5 or all)                                 # Required
    --mixed                                                           # Indicate if you are predicting from a co-culture                                                                # Not required
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS] \                         # How many ROIs of each target class are included in the training set                                             # Required
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY]                   # Seed for data sampling and splitting in train/test/validation sets                                              # Default '0'
```

## 4) Evaluation
```
python embeddings.py \                                                # Script for extracting the feature embeddings from a trained CNN model
    --input_path [PATH_TO_CP_CROPS] \                                 # Input path to the CP crops you want to extract embeddings from                # Required
    --output_path [OUTPUT_PATH] \                                     # Output path where you want to store the .csv file with embeddings             # Required
    --model_file [PATH_TO_MODEL_FILE] \                               # Path to the trained CNN network (.pth file)                                   # Required
    --target_names [CULTURE_NAMES_TO_TRAIN_ON] \                      # Names of the prediction classes (as in the layout file)                       # Required
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \                 # Path to the layout file (excel), reflecting the contents of the plate layout  # Required
    --GT_data_file [PATH_TO_GT_FILE] \                                # .csv file containing the true phenotype of each ROI                           # Required
    --image_size [IMAGES_ARE_RESIZED_TO_THIS_SIZE | 128] \            # Size (in pixels) to which the input crops are resized for training            # Default '128' 
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY] \                 # Seed for data sampling and splitting in train/test/validation sets            # Default '0'  
    --mixed                                                           # Indicate if you are predicting from a co-culture                              # Not required
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS]                           # How many ROIs of each target class are included in the training set           # Required
```
```
python grad_cam.py \                                                  # Script for generating GradCAM heatmaps of individual crops
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

(1) Download the test dataset

# HOW TO CITE
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
