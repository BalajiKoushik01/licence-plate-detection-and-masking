# Troubleshooting Guide

This guide provides solutions to common issues you might encounter when using the Vehicle Detection System.

## Installation Issues

### Package Installation Fails

**Problem**: Error when installing the package with pip.

**Solution**:
1. Ensure you have the latest version of pip:
   ```bash
   python -m pip install --upgrade pip
   ```

2. Check that you have the required dependencies for building packages:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install python3-dev build-essential
   
   # On Windows
   # Install Visual C++ Build Tools from https://visualstudio.microsoft.com/visual-cpp-build-tools/
   ```

3. Try installing with the `--no-cache-dir` option:
   ```bash
   pip install --no-cache-dir vehicle-detection
   ```

### Import Errors

**Problem**: `ImportError` or `ModuleNotFoundError` when importing the package.

**Solution**:
1. Verify the package is installed:
   ```bash
   pip list | grep vehicle-detection
   ```

2. Check your Python path:
   ```python
   import sys
   print(sys.path)
   ```

3. If you installed from source, ensure you installed in development mode:
   ```bash
   pip install -e .
   ```

## Model Loading Issues

### Model File Not Found

**Problem**: Error message about the model file not being found.

**Solution**:
1. Check if the model file exists in the expected location:
   ```bash
   # Default location is the current working directory
   ls yolov8n.pt
   ```

2. Download the model manually:
   ```bash
   # Using wget
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   
   # Using curl
   curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -o yolov8n.pt
   ```

3. Specify the full path to the model file:
   ```python
   from vehicle_detection.detector import load_model
   
   model = load_model("/full/path/to/yolov8n.pt")
   ```

### CUDA/GPU Issues

**Problem**: Errors related to CUDA or GPU acceleration.

**Solution**:
1. Check if CUDA is available:
   ```python
   import torch
   print(torch.cuda.is_available())
   print(torch.cuda.device_count())
   print(torch.cuda.get_device_name(0))
   ```

2. Install the correct version of PyTorch for your CUDA version:
   ```bash
   # Visit https://pytorch.org/get-started/locally/ for installation instructions
   ```

3. If you don't have a GPU or don't want to use it, force CPU mode:
   ```python
   import os
   os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
   ```

## Detection Issues

### No Vehicles Detected

**Problem**: The system doesn't detect any vehicles in images where vehicles are clearly present.

**Solution**:
1. Lower the confidence threshold:
   ```bash
   vehicle-detect -i input.jpg -o output --conf 0.1
   ```

2. Check if the image format is supported:
   ```bash
   # Convert the image to a common format like JPEG
   from PIL import Image
   img = Image.open("input.png")
   img.save("input.jpg")
   ```

3. Ensure the image is not corrupted:
   ```python
   import cv2
   img = cv2.imread("input.jpg")
   if img is None:
       print("Image is corrupted or in an unsupported format")
   ```

### Poor Detection Quality

**Problem**: Vehicles are detected but with poor accuracy or many false positives.

**Solution**:
1. Adjust the confidence threshold:
   ```bash
   # Increase for fewer false positives
   vehicle-detect -i input.jpg -o output --conf 0.4
   
   # Decrease for fewer false negatives
   vehicle-detect -i input.jpg -o output --conf 0.2
   ```

2. Try a different YOLOv8 model:
   ```bash
   # Download a larger model for better accuracy
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
   
   # Use the model in your code
   from vehicle_detection.detector import load_model, detect_vehicles
   
   model = load_model("yolov8m.pt")
   detect_vehicles("input.jpg", "output.jpg")
   ```

## Output Issues

### Permission Errors When Saving Output

**Problem**: Permission errors when trying to save output images.

**Solution**:
1. Check if the output directory exists and is writable:
   ```bash
   # Create the output directory if it doesn't exist
   mkdir -p output
   
   # Check permissions
   ls -la output
   ```

2. Specify a different output directory:
   ```bash
   vehicle-detect -i input.jpg -o /path/to/writable/directory
   ```

### Output Image Not Opening

**Problem**: The output image is saved but doesn't open automatically.

**Solution**:
1. Check if the image was saved correctly:
   ```bash
   ls -la output
   ```

2. Open the image manually with your preferred image viewer.

3. Disable automatic opening:
   ```bash
   vehicle-detect -i input.jpg -o output --open False
   ```

## Performance Issues

### Slow Processing

**Problem**: Image processing is very slow.

**Solution**:
1. Use a smaller YOLOv8 model:
   ```bash
   # Download a smaller model
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   ```

2. Process smaller images:
   ```python
   import cv2
   from vehicle_detection.detector import detect_vehicles
   
   # Resize the image before processing
   img = cv2.imread("input.jpg")
   img_resized = cv2.resize(img, (640, 480))
   cv2.imwrite("input_resized.jpg", img_resized)
   
   detect_vehicles("input_resized.jpg", "output.jpg")
   ```

3. Use GPU acceleration if available:
   ```bash
   # Install PyTorch with CUDA support
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

## Command-Line Interface Issues

### Command Not Found

**Problem**: `vehicle-detect` command not found after installation.

**Solution**:
1. Ensure the package is installed:
   ```bash
   pip install vehicle-detection
   ```

2. Check if the entry point script is in your PATH:
   ```bash
   which vehicle-detect  # On Unix/Linux/macOS
   where vehicle-detect  # On Windows
   ```

3. If installed in a virtual environment, ensure it's activated:
   ```bash
   source venv/bin/activate  # On Unix/Linux/macOS
   venv\Scripts\activate     # On Windows
   ```

### Invalid Arguments

**Problem**: Error about invalid command-line arguments.

**Solution**:
1. Check the help message for correct usage:
   ```bash
   vehicle-detect --help
   ```

2. Ensure file paths are correct and exist:
   ```bash
   # Check if input file exists
   ls input.jpg
   ```

3. Use quotes for paths with spaces:
   ```bash
   vehicle-detect -i "path with spaces/input.jpg" -o "output dir"
   ```

## Still Having Issues?

If you're still experiencing problems after trying these solutions:

1. Check the log file for more detailed error messages:
   ```bash
   cat vehicle_detection.log
   ```

2. Open an issue on GitHub with:
   - A detailed description of the problem
   - Steps to reproduce the issue
   - Error messages from the log file
   - Information about your environment (OS, Python version, etc.)

3. Join the community discussion on [GitHub Discussions](https://github.com/username/vehicle_detection/discussions) for help from other users.