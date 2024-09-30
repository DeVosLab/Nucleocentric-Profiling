# Unbiased identification of cell identity in dense mixed neural cultures
Link to ELife paper: https://elifesciences.org/reviewed-preprints/95273v1#s4

Each folder within the repository groups the source code related to distinct steps explained below. The util file contains functions that can be called upon.


![image](https://github.com/user-attachments/assets/279e1514-c83f-451b-ab3d-22a33f56bbf3)


## 1) Segmentation
Identification of ROIs in the CP image.

**Full cell segmentation**
```
python segment_cells.py \
    -i [INPUT_PATH] \
    -o [OUTPUT_PATH] \
    --file_extension [FILE_EXTENSION] \
    --save_masks \
    --pmin 1 --pmax 99 \
    --gpu --net_avg \
    --channels2use 0 1 2 3 \
    --channels2store 0 1 2 3 \
    --target cyto \
    --diameter 100
```

**Nuclei segmentation** 
```
python segment_nuclei.py \
    -i [INPUT_PATH] \
    -o [OUTPUT_PATH] \
    --file_extension [FILE_EXTENSION] \
    --pmin 1 --pmax 99 \
    --gpu \
    --probability 0.6 --overlap 0.03

```


## 2) Processing
Operations performed on the whole image. 
GT2CP alignment: necessary for mixed cultures where post-hoc staining was performed to identify the true cell phenotype. The GT image is translated to overlap with the CP image and masks identified in step 1.
Cropping of ROIs: individual cells (ROIs) are cropped out of the original image. Patch size (in pixel) for full cell = 192, for nucleus and nucleocentric crops: 60. Masked patch = True for full cell and nucleus.
Get intensity and texture features: handcrafted features extracted for RF and UMAP.

**GT alignment** 
```
python align_GT2PC.py \
    --CP_path [PATH_TO_CP_IMAGES] \
    --GT_path [PATH_TO_GT_IMAGS] \
    --GT_name [NAME_OF_STAINING] \
    --masks_path [PATH_TO_MASKS] \
    --channels2use_CP [CP_CHANNEL_IDX_TO_BE_USED_FOR_ALIGNMENT] \
    --channels2use_GT [GT_CHANNEL_IDX_TO_BE_USED_FOR_ALIGNMENT] \
    --file_extension_imgs [FILE_EXTENSION_CP_AND_GT_IMAGES] \
    --file_extension_masks [FILE_EXTENSION_MASKS] \
    --output_path [OUTPUT_PATH]
```



**Cropping of ROIs** 
```
python crop_ROIs.py \
    --CP_path [PATH_TO_CP_IMAGES] \
    --masks_path [PATH_TO_MASKS] \
    --file_extension_imgs [FILE_EXTENSION_CP_AND_GT_IMAGES] \
    --file_extension_masks [FILE_EXTENSION_MASKS] \
    --output_path [OUTPUT_PATH] \
    --masked_patch \
    --patch_size 192
```
**Handcrafted feature extraction** 
```
python get_intensity_features.py \
    --GT_path [PATH_TO_IMAGES] \
    --GT_channel_names [NAMES_OF_CHANNELS] \
    --masks_path [PATH_TO_MASKS] \
    --layout [PATH_TO_LAYOUT_FILE] \
    --file_extension_GT [FILE_EXTENSION_IMAGES] \
    --file_extension_masks [FILE_EXTENSION_MASKS] \
    --output_path [OUTPUT_PATH] \
    --masked_patch \
    --patch_size 192
```

```
python get_texture_features.py \
    --CP_path [PATH_TO_IMAGES] \
    --masks_path [PATH_TO_MASKS] \
    --file_extension_imgs [FILE_EXTENSION_CP_AND_GT_IMAGES] \
    --file_extension_masks [FILE_EXTENSION_MASKS] \
    --output_path [OUTPUT_PATH] \
    --masked_patch \
    --patch_size 192
```

Example of ground-truth identification (with intensity features extracted using 'get_intensity_features.py')
![image](https://github.com/user-attachments/assets/be65a0e6-8fe3-435e-ab98-f748bf611c41)


## 3) Classification

```
python train_evaluate_CNN.py \
    --input_path [PATH_TO_CP_CROPS] \
    --output_path [OUTPUT_PATH] \
    --target_names [CULTURE_NAMES_TO_TRAIN_ON] \
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \
    --GT_data_file [PATH_TO_GT_FILE] \
    --epochs [NUMBER_OF_TRAINING_EPOCHS | 50] \
    --image_size [IMAGES_ARE_RESIZED_TO_THIS_SIZE | 128] \
    --channels2use [CHANNELS_USED_IN_TRAINING] \
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY] \
    --save \
    --mixed
    --batch_size [NUMBER_OF_SAMPLES_TO_PROCESS_PER_ITERATION | 100] \
    --learning_rate [LEARNING_RATE | 0.0001]
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS]
```

```
python train_evaluate_RF.py \
    --data_file [PATH_TO_FEATURE_DATA] \
    --regions2use [REGIONS_AS_INPUT] \
    --channels2use [CHANNELS_AS_INPUT] \
    --mixed
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS] \
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY]
```

## 4) Evaluation
```
python embeddings.py \
    --input_path [PATH_TO_CP_CROPS] \
    --output_path [OUTPUT_PATH] \
    --model_file [PATH_TO_MODEL_FILE] \
    --target_names [CULTURE_NAMES_TO_TRAIN_ON] \
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \
    --GT_data_file [PATH_TO_GT_FILE] \
    --image_size [IMAGES_ARE_RESIZED_TO_THIS_SIZE | 128] \
    --channels2use [CHANNELS_USED] \
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY] \
    --mixed
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS]
```
```
python grad_cam.py \
    --input_path [PATH_TO_CP_CROPS] \
    --model_file [PATH_TO_MODEL_FILE] \
    --target_names [CULTURE_NAMES_TO_TRAIN_ON] \
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \
    --GT_data_file [PATH_TO_GT_FILE] \
    --image_size [IMAGES_ARE_RESIZED_TO_THIS_SIZE | 128] \
    --channels2use [CHANNELS_USED] \
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY] \
    --mixed
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS]
```
```
python plate_overview.py \
    --input_path [PATH_TO_IMAGES] \
    --output_path [OUTPUT_PATH] \
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \
    --file_extension [FILE_EXTENSION_OF_IMAGES] \
    --pmin 1 --pmax 99 \
    --gpu \
    --channels2use [CHANNELS_USED]
```
```
python predict_CNN.py \
    --input_path [PATH_TO_CP_CROPS] \
    --model_file [PATH_TO_MODEL_FILE] \
    --target_names [CULTURE_NAMES_TO_TRAIN_ON] \
    --layout [PATH_TO_EXCEL_FILE_WITH_PLATE_LAYOUT] \
    --GT_data_file [PATH_TO_GT_FILE] \
    --image_size [IMAGES_ARE_RESIZED_TO_THIS_SIZE | 128] \
    --channels2use [CHANNELS_USED] \
    --random_seed [RANDOM_SEED_FOR_REPRODUCIBILITY] \
    --mixed
    --sample [SAMPLE_SIZE_PER_TARGET_CLASS]
```

# CITATION
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
