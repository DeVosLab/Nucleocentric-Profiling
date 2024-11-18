"""
Nucleocentric-profiling: Cell identity classification in mixed neural cultures
"""

# Core functionality imports
from nucleocentric.segmentation.segment_cells import segment_cells
from nucleocentric.segmentation.segment_nuclei import segment_nuclei
from nucleocentric.preprocessing.alignment import align_GT2CP
from nucleocentric.preprocessing.cropping import crop_ROIs
from nucleocentric.train.train_cnn import train_model as train_CNN
from nucleocentric.train.train_rf import train_model as train_RF
from scripts.grad_cam import generate_gradcam
from scripts.UMAP import generate_umap

# Version info
__version__ = '0.1.0'
__author__ = 'Sarah De Beuckeleer et al.'

# Package metadata
__all__ = [
    # Segmentation
    'segment_cells',
    'segment_nuclei',
    
    # Preprocessing
    'align_GT2CP',
    'crop_ROIs',
    
    # Feature extraction
    'get_intensity_features',
    'get_texture_features',
    
    # Model training
    'train_CNN',
    'train_RF',
    
    # Evaluation
    'generate_gradcam',
    'generate_umap'
]