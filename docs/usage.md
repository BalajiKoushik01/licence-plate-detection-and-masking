# Usage Guide

This guide provides examples and instructions for using the Vehicle Detection System.

## Command-Line Interface

The Vehicle Detection System provides a command-line interface for detecting vehicles in images.

### Basic Usage

To detect vehicles in a single image:

```bash
vehicle-detect -i path/to/image.jpg -o path/to/output
```

This will process the image, detect vehicles, and save the output image with bounding boxes to the specified output directory.

### Command-Line Options

| Option | Description |
|--------|-------------|
| `-i, --input` | Path to input image or directory |
| `-o, --output` | Path to output directory |
| `--batch` | Process all images in the input directory |
| `--show` | Display detection results |
| `--open` | Open images after saving (default: True) |
| `--conf` | Confidence threshold (0.0-1.0, default: 0.25) |
| `--random-colors` | Use random colors for different vehicle types |
| `--thickness` | Line thickness for bounding boxes (default: 2) |
| `--font-scale` | Font scale for labels (default: 0.6) |

### Examples

Process a single image with a higher confidence threshold:

```bash
vehicle-detect -i inputs/car1.jpg -o output --conf 0.3
```

Process all images in a directory:

```bash
vehicle-detect -i inputs --output results --batch
```

Use random colors and display results:

```bash
vehicle-detect -i inputs/bus1.jpg --random-colors --show
```

## Using as a Python Module

You can also use the Vehicle Detection System as a Python module in your own code.

### Detecting Vehicles in a Single Image

```python
from vehicle_detection.detector import detect_vehicles

# Detect vehicles in a single image
boxes, counts = detect_vehicles(
    image_path="path/to/image.jpg",
    save_path="path/to/output.jpg",
    conf_threshold=0.3
)

# Print detection results
print(f"Detected {counts['Total']} vehicles:")
for vehicle_type, count in counts.items():
    if vehicle_type != "Total" and count > 0:
        print(f"  - {vehicle_type}: {count}")
```

### Processing Multiple Images

```python
from vehicle_detection.detector import batch_process

# Process all images in a directory
results = batch_process(
    input_dir="path/to/images",
    output_dir="path/to/output",
    conf_threshold=0.3
)

# Print summary
total_vehicles = sum(counts.get("Total", 0) for counts in results.values())
print(f"Processed {len(results)} images, detected {total_vehicles} vehicles total.")
```

### Customizing Colors

```python
from vehicle_detection.detector import detect_vehicles

# Define custom colors for different vehicle types
custom_colors = {
    "Car": (0, 255, 0),       # Green
    "Motorcycle": (0, 0, 255), # Red
    "Bus": (255, 0, 0),       # Blue
    "Truck": (255, 255, 0),   # Cyan
}

# Detect vehicles with custom colors
detect_vehicles(
    image_path="path/to/image.jpg",
    save_path="path/to/output.jpg",
    custom_colors=custom_colors
)
```

## Advanced Usage

### Adjusting Detection Parameters

You can adjust various parameters to fine-tune the detection:

```python
from vehicle_detection.detector import detect_vehicles

detect_vehicles(
    image_path="path/to/image.jpg",
    save_path="path/to/output.jpg",
    conf_threshold=0.4,        # Higher confidence threshold
    line_thickness=3,          # Thicker bounding boxes
    font_scale=0.8             # Larger font for labels
)
```

### Handling Detection Results

The `detect_vehicles` function returns a tuple containing:
1. A list of detected vehicle boxes
2. A dictionary with vehicle counts by type

You can use these results for further processing:

```python
from vehicle_detection.detector import detect_vehicles

boxes, counts = detect_vehicles(
    image_path="path/to/image.jpg",
    save_path="path/to/output.jpg"
)

# Access individual detections
for box in boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
    conf = float(box.conf[0])               # Confidence score
    cls = int(box.cls[0])                   # Class ID
    
    # Do something with the detection
    print(f"Vehicle detected at ({x1}, {y1}, {x2}, {y2}) with confidence {conf:.2f}")
```

## Troubleshooting

If you encounter issues while using the Vehicle Detection System, please check the following:

1. Ensure the input image exists and is a valid image file
2. Check that the output directory is writable
3. Verify that the YOLOv8 model file is available
4. Check the log file (`vehicle_detection.log`) for error messages

For more help, please refer to the [installation guide](installation.md) or [open an issue](https://github.com/username/vehicle_detection/issues) on GitHub.