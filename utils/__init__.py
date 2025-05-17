"""
Vehicle Detection Utilities
==========================

This package provides utility functions for vehicle detection and license plate masking.
"""

from utils.car_detector import detect_cars, batch_process, get_detector, generate_random_colors, load_model
from utils.plate_utils import mask_plate_with_logo

__all__ = [
    'detect_cars',
    'batch_process',
    'load_model',
    'generate_random_colors',
    'mask_plate_with_logo'
]