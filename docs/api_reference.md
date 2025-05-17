# API Reference

This document provides detailed information about the functions and classes in the Vehicle Detection System.

## Module: vehicle_detection.detector

This module provides the core functionality for detecting vehicles in images.

### Constants

#### VEHICLE_CLASSES

A dictionary mapping class IDs to vehicle type names.

```python
VEHICLE_CLASSES = {
    2: "Car", 
    3: "Motorcycle", 
    5: "Bus", 
    7: "Truck"
}
```

#### DEFAULT_COLORS

A dictionary mapping vehicle types to RGB color tuples for visualization.

```python
DEFAULT_COLORS = {
    "Car": (0, 255, 0),       # Green
    "Motorcycle": (0, 165, 255),  # Orange
    "Bus": (255, 0, 0),       # Blue
    "Truck": (0, 0, 255),     # Red
    "Vehicle": (255, 255, 0)  # Cyan (fallback)
}
```

### Functions

#### load_model

```python
load_model(model_path: str = "yolov8n.pt", timeout: int = 30) -> Optional[Any]
```

Load the YOLO model with timeout and error handling.

**Parameters:**
- `model_path` (str): Path to the model file. Default is "yolov8n.pt".
- `timeout` (int): Maximum time to wait for model loading in seconds. Default is 30.

**Returns:**
- YOLO model or None if loading fails.

#### generate_random_colors

```python
generate_random_colors(num_colors: int = 5) -> Dict[str, Tuple[int, int, int]]
```

Generate random colors for visualization.

**Parameters:**
- `num_colors` (int): Number of colors to generate. Default is 5.

**Returns:**
- Dictionary mapping class names to RGB color tuples.

#### detect_vehicles

```python
detect_vehicles(
    image_path: str, 
    save_path: Optional[str] = None, 
    show: bool = False,
    conf_threshold: float = 0.25,
    custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    open_image: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.6
) -> Tuple[List, Dict[str, int]]
```

Detect vehicles in an image.

**Parameters:**
- `image_path` (str): Path to the input image.
- `save_path` (Optional[str]): Path to save the output image. Default is None.
- `show` (bool): Whether to display the image with detections. Default is False.
- `conf_threshold` (float): Confidence threshold for detections. Default is 0.25.
- `custom_colors` (Optional[Dict[str, Tuple[int, int, int]]]): Custom colors for different vehicle types. Default is None.
- `open_image` (bool): Whether to open the saved image. Default is True.
- `line_thickness` (int): Thickness of bounding box lines. Default is 2.
- `font_scale` (float): Scale of the font for labels. Default is 0.6.

**Returns:**
- Tuple containing list of vehicle boxes and dictionary with vehicle counts by type.

#### batch_process

```python
batch_process(
    input_dir: str, 
    output_dir: str, 
    conf_threshold: float = 0.25,
    custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    extensions: List[str] = ['.jpg', '.jpeg', '.png'],
    show: bool = False,
    open_images: bool = False
) -> Dict[str, Dict[str, int]]
```

Process all images in a directory.

**Parameters:**
- `input_dir` (str): Directory containing input images.
- `output_dir` (str): Directory to save output images.
- `conf_threshold` (float): Confidence threshold for detections. Default is 0.25.
- `custom_colors` (Optional[Dict[str, Tuple[int, int, int]]]): Custom colors for different vehicle types. Default is None.
- `extensions` (List[str]): List of valid file extensions to process. Default is ['.jpg', '.jpeg', '.png'].
- `show` (bool): Whether to display images with detections. Default is False.
- `open_images` (bool): Whether to open saved images. Default is False.

**Returns:**
- Dictionary mapping filenames to vehicle counts.

## Module: vehicle_detection.main

This module provides the command-line interface for the Vehicle Detection System.

### Functions

#### parse_arguments

```python
parse_arguments()
```

Parse command line arguments.

**Returns:**
- Parsed command-line arguments.

#### process_single_image

```python
process_single_image(args) -> Tuple[int, Dict]
```

Process a single image with the given arguments.

**Parameters:**
- `args`: Command-line arguments.

**Returns:**
- Tuple containing number of detected vehicles and counts by type.

#### main

```python
main() -> int
```

Main function for the command-line interface.

**Returns:**
- Exit code (0 for success, 1 for error).

#### main_cli

```python
main_cli()
```

Entry point for the command-line interface.