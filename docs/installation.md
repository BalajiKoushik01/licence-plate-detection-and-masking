# Installation Guide

This guide provides instructions for installing the Vehicle Detection System.

## Prerequisites

Before installing the Vehicle Detection System, ensure you have the following prerequisites:

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)

## Installation Methods

### Method 1: Install from PyPI (Recommended)

The simplest way to install the Vehicle Detection System is via pip:

```bash
pip install vehicle-detection
```

This will install the package and all its dependencies.

### Method 2: Install from Source

If you want to install the latest development version or contribute to the project, you can install from source:

1. Clone the repository:
   ```bash
   git clone https://github.com/username/vehicle_detection.git
   cd vehicle_detection
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Installing Dependencies Manually

If you prefer to install dependencies manually, you can use the requirements.txt file:

```bash
pip install -r requirements.txt
```

The main dependencies are:
- ultralytics (for YOLOv8)
- opencv-python
- numpy

## Verifying Installation

To verify that the installation was successful, run:

```bash
python -c "import vehicle_detection; print(vehicle_detection.__version__)"
```

This should print the version number of the installed package.

## Downloading the YOLOv8 Model

The Vehicle Detection System uses the YOLOv8 model for object detection. The model will be downloaded automatically the first time you run the system. However, if you want to download it manually, you can do so from the [Ultralytics website](https://github.com/ultralytics/ultralytics).

## Troubleshooting

If you encounter any issues during installation, please check the following:

1. Ensure you have the correct Python version:
   ```bash
   python --version
   ```

2. Ensure pip is up to date:
   ```bash
   pip install --upgrade pip
   ```

3. If you're having issues with OpenCV, try installing it separately:
   ```bash
   pip install opencv-python
   ```

4. If you're still having issues, please [open an issue](https://github.com/username/vehicle_detection/issues) on GitHub.