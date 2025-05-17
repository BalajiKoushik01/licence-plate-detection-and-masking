"""
Vehicle Detection CLI
====================

This module provides a command-line interface for the vehicle detection system.

It includes:
- Command-line argument parsing
- Single image processing
- Batch processing of directories
<<<<<<< HEAD
- Real-time webcam processing
- Error handling and logging
- Database logging for detections
- Analytics and visualization
=======
- Error handling and logging
>>>>>>> origin/main
"""

import sys
import csv
<<<<<<< HEAD
import subprocess
import platform
import cv2
import os
import logging
import time
import argparse
import json
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from multiprocessing import Pool
from typing import Dict, Tuple, List, Optional
import xml.etree.ElementTree as ET
from tqdm import tqdm
import torch
import numpy as np

# Import detection and processing functions
from vehicle_detection.plate_recognizer import extract_license_plates
from vehicle_detection.video_processor import process_video, real_time_detection
from vehicle_detection.detector import detect_vehicles, generate_random_colors, DEFAULT_COLORS, VehicleDetector
=======
import cv2
import os
import argparse
import logging
import time
from typing import Dict, Tuple
>>>>>>> origin/main

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vehicle_detection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

<<<<<<< HEAD
# Define the path to the logo image
DEFAULT_LOGO_PATH = "assets/logo.png"

def resolve_logo_path(logo_path: str) -> Optional[str]:
    """
    Resolve the logo path by checking multiple locations.

    Args:
        logo_path: Initial logo path to resolve.

    Returns:
        Resolved logo path or None if not found.
    """
    if not logo_path:
        return None

    # Check if the path is absolute and exists
    if os.path.isabs(logo_path) and os.path.exists(logo_path):
        return logo_path

    # Resolve relative path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    possible_paths = [
        logo_path,
        os.path.join(project_root, logo_path),
        os.path.join(os.path.dirname(project_root), logo_path),
        os.path.join(os.path.dirname(project_root), "assets", "logo.png")
    ]

    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"[INFO] Found logo at: {path}")
            return os.path.abspath(path)

    logger.warning(f"[WARNING] Logo image not found in any location: {logo_path}")
    return None

def validate_logo_path(logo_path: Optional[str]) -> Optional[str]:
    """
    Validate the logo image and prompt the user if necessary.

    Args:
        logo_path: Path to the logo image.

    Returns:
        Validated logo path or None if invalid.
    """
    if not logo_path:
        logger.info("[INFO] No logo path provided. Plates will be masked with a solid color.")
        return None

    logo_path = resolve_logo_path(logo_path)
    if not logo_path:
        print(f"[WARNING] Logo image not found: {logo_path}")
        print("Would you like to:")
        print("1. Provide a path to a different logo image")
        print("2. Continue without logo (number plates will be masked with a solid color)")
        choice = input("Enter your choice (1 or 2): ")

        if choice == "1":
            new_logo_path = input("Enter the path to the logo image: ")
            logo_path = resolve_logo_path(new_logo_path)
            if logo_path:
                logger.info(f"[INFO] Using logo from: {logo_path}")
            else:
                logger.warning(f"[WARNING] Logo image not found at: {new_logo_path}")
                print("Continuing without logo. Number plates will be masked with a solid color.")
                logo_path = None
        else:
            logger.info("[INFO] Continuing without logo. Number plates will be masked with a solid color.")
            print("Number plates will be masked with a solid color.")
            logo_path = None

    if logo_path:
        try:
            logo_image = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo_image is None:
                logger.warning("[WARNING] Logo image is not a valid image format.")
                print("Continuing without logo. Number plates will be masked with a solid color.")
                logo_path = None
            else:
                logger.info(f"[SUCCESS] Logo image loaded successfully from: {logo_path}")
        except Exception as e:
            logger.warning(f"[WARNING] Failed to read logo image: {e}")
            print("Continuing without logo. Number plates will be masked with a solid color.")
            logo_path = None

    return logo_path

def init_db():
    """
    Initialize SQLite database for logging detections.

    Returns:
        SQLite connection object.
    """
    try:
        conn = sqlite3.connect("vehicles.db")
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS vehicles
                     (timestamp TEXT, vehicle_type TEXT, license_plate TEXT, confidence REAL,
                      bbox TEXT, image_path TEXT)''')
        conn.commit()
        logger.info("[INFO] Database initialized successfully.")
        return conn
    except sqlite3.Error as e:
        logger.error(f"[ERROR] Failed to initialize database: {e}")
        raise

def log_vehicle(conn, vehicle_type: str, license_plate: str, confidence: float, bbox: Tuple[int, int, int, int], image_path: str):
    """
    Log a detected vehicle to the database.

    Args:
        conn: SQLite connection object.
        vehicle_type: Type of the vehicle.
        license_plate: Detected license plate text.
        confidence: Detection confidence score.
        bbox: Bounding box coordinates.
        image_path: Path to the processed image.
    """
    try:
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        bbox_str = json.dumps(bbox)
        c.execute("INSERT INTO vehicles VALUES (?, ?, ?, ?, ?, ?)",
                  (timestamp, vehicle_type, license_plate, confidence, bbox_str, image_path))
        conn.commit()
        logger.debug(f"[DEBUG] Logged vehicle: {vehicle_type}, Plate: {license_plate}, Confidence: {confidence}")
    except sqlite3.Error as e:
        logger.error(f"[ERROR] Failed to log vehicle to database: {e}")

def generate_report(vehicle_counts: Dict[str, int], plates: List[str], output_dir: str, process_time: float):
    """
    Generate a textual report of detection results.

    Args:
        vehicle_counts: Dictionary of vehicle counts by type.
        plates: List of detected license plates.
        output_dir: Directory to save the report.
        process_time: Total processing time.
    """
    try:
        report_path = os.path.join(output_dir, "detection_report.txt")
        os.makedirs(output_dir, exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Vehicle Detection Report\n")
            f.write("=======================\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Processing Time: {process_time:.2f} seconds\n\n")
            f.write("Vehicle Counts:\n")
            for vehicle_type, count in vehicle_counts.items():
                f.write(f"  - {vehicle_type}: {count}\n")
            f.write("\nDetected License Plates:\n")
            if plates:
                for plate in plates:
                    f.write(f"  - {plate}\n")
            else:
                f.write("  None\n")
        logger.info(f"[SUCCESS] Report generated at: {report_path}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate report: {e}")

def plot_results(vehicle_counts: Dict[str, int], plates: List[str], output_dir: str):
    """
    Visualize vehicle counts and plate detection frequency using plots.

    Args:
        vehicle_counts: Dictionary of vehicle counts by type.
        plates: List of detected license plates.
        output_dir: Directory to save the plots.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)

        # Plot vehicle counts
        plt.figure(figsize=(8, 6))
        vehicle_types = [k for k in vehicle_counts.keys() if k != "Total"]
        counts = [vehicle_counts[k] for k in vehicle_types]
        plt.bar(vehicle_types, counts)
        plt.title("Vehicle Counts by Type")
        plt.xlabel("Vehicle Type")
        plt.ylabel("Count")
        plt.xticks(rotation=45)
        vehicle_plot_path = os.path.join(output_dir, "vehicle_counts.png")
        plt.savefig(vehicle_plot_path)
        plt.close()
        logger.info(f"[SUCCESS] Vehicle counts plot saved to: {vehicle_plot_path}")

        # Plot plate detection frequency
        if plates:
            plt.figure(figsize=(8, 6))
            plate_freq = {}
            for plate in plates:
                plate_freq[plate] = plate_freq.get(plate, 0) + 1
            plt.bar(list(plate_freq.keys()), list(plate_freq.values()))
            plt.title("License Plate Detection Frequency")
            plt.xlabel("Plate Text")
            plt.ylabel("Frequency")
            plt.xticks(rotation=45)
            plate_plot_path = os.path.join(output_dir, "plate_frequency.png")
            plt.savefig(plate_plot_path)
            plt.close()
            logger.info(f"[SUCCESS] Plate frequency plot saved to: {plate_plot_path}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to generate plots: {e}")

def preprocess_image(image: np.ndarray, max_size: int = 640) -> np.ndarray:
    """
    Preprocess image by resizing to reduce computation.

    Args:
        image: Input image.
        max_size: Maximum dimension for resizing.

    Returns:
        Preprocessed image.
    """
    h, w = image.shape[:2]
    scale = min(max_size / h, max_size / w)
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(image, (new_w, new_h))

def parse_arguments():
    """
    Parse command line arguments with enhanced options.

    Returns:
        Parsed command-line arguments.
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_input = os.path.join(project_root, "inputs")
    default_output = os.path.join(project_root, "output")

    # Resolve default input directory
    if not os.path.exists(default_input):
        parent_input = os.path.join(os.path.dirname(project_root), "inputs")
        if os.path.exists(parent_input):
            default_input = parent_input
            logger.info(f"[INFO] Found inputs directory at parent level: {default_input}")

    # Resolve default output directory
    if not os.path.exists(default_output):
        parent_output = os.path.join(os.path.dirname(project_root), "output")
        if os.path.exists(parent_output):
            default_output = parent_output
            logger.info(f"[INFO] Found output directory at parent level: {default_output}")
        else:
            try:
                os.makedirs(default_output, exist_ok=True)
                logger.info(f"[INFO] Created output directory: {default_output}")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to create output directory: {e}")
                default_output = os.path.join(project_root, "outputs")

    parser = argparse.ArgumentParser(description="Vehicle Detection Tool")
    parser.add_argument("-i", "--input", type=str, default=default_input,
                        help="Path to input image, video, or directory")
    parser.add_argument("-o", "--output", type=str, default=default_output,
                        help="Path to output directory")

    # Processing options
    parser.add_argument("--batch", action="store_true",
                        help="Process all images and videos in the input directory")
    parser.add_argument("--video", action="store_true",
                        help="Process input as a video file")
    parser.add_argument("--webcam", action="store_true",
                        help="Perform real-time detection using webcam")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0.0-1.0)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for detection if available")
    parser.add_argument("--plate-model", type=str, default="yolov8_plate.pt",
                        help="Path to the fine-tuned YOLOv8 model for plate detection")
    parser.add_argument("--region", type=str, default="India",
                        help="Region for plate text validation (e.g., India, US, EU)")
    parser.add_argument("--ocr-engine", type=str, choices=["easyocr", "tesseract"], default="easyocr",
                        help="OCR engine to use for license plate recognition")
=======
# Import vehicle detection functions
try:
    from vehicle_detection.detector import (
        detect_vehicles, 
        batch_process, 
        generate_random_colors, 
        DEFAULT_COLORS
    )
    logger.info("[SUCCESS] Vehicle detection modules successfully imported.")
except Exception as e:
    logger.error(f"[ERROR] Import failed: {str(e)}")
    sys.exit(1)


def parse_arguments():
    """
    Parse command line arguments

    Returns:
        Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description="Vehicle Detection Tool")

    # Input/output options
    parser.add_argument("-i", "--input", type=str, default="inputs/bus1.jpg",
                        help="Path to input image or directory")
    parser.add_argument("-o", "--output", type=str, default="output",
                        help="Path to output directory")

    # Processing options
    parser.add_argument("--batch", action="store_true", 
                        help="Process all images in the input directory")
    parser.add_argument("--show", action="store_true", 
                        help="Display detection results")
    parser.add_argument("--open", action="store_true", default=True,
                        help="Open images after saving")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (0.0-1.0)")
>>>>>>> origin/main

    # Visualization options
    parser.add_argument("--random-colors", action="store_true",
                        help="Use random colors for different vehicle types")
    parser.add_argument("--thickness", type=int, default=2,
                        help="Line thickness for bounding boxes")
    parser.add_argument("--font-scale", type=float, default=0.6,
                        help="Font scale for labels")
<<<<<<< HEAD
    parser.add_argument("--detect-plates", action="store_true", default=True,
                        help="Detect license plates after vehicle detection")
    parser.add_argument("--replace-logo", action="store_true", default=True,
                        help="Replace plates with logo")
    parser.add_argument("--photorealistic", action="store_true",
                        help="Use photorealistic logo overlay with lighting and shadows")
    parser.add_argument("--logo", type=str, default=DEFAULT_LOGO_PATH,
                        help="Path to logo image")

    # Video processing options
    parser.add_argument("--video-fps", type=int, default=30,
                        help="Frames per second for video processing")
    parser.add_argument("--video-output", type=str, default="output/video_output.mp4",
                        help="Path to save video output")
    parser.add_argument("--video-crop-dir", type=str, default="output/cropped_vehicles",
                        help="Directory to save cropped vehicles")
    parser.add_argument("--video-plates-dir", type=str, default="output/plates",
                        help="Directory to save detected plates")

    # Reporting options
    parser.add_argument("--save-csv", action="store_true",
                        help="Save detection data to CSV")
    parser.add_argument("--save-json", action="store_true",
                        help="Save detection data to JSON")
    parser.add_argument("--save-xml", action="store_true",
                        help="Save detection data to XML")
    parser.add_argument("--generate-report", action="store_true",
                        help="Generate a textual report of detection results")
    parser.add_argument("--plot-results", action="store_true",
                        help="Generate plots of vehicle counts and plate frequency")

    return parser.parse_args()

def process_single_image(args) -> Tuple[int, Dict[str, int], List[str]]:
    """
    Process a single image with the given arguments, detecting vehicles and license plates.

    Args:
        args: Command-line arguments.

    Returns:
        Tuple containing number of detected vehicles, vehicle counts by type, and detected plates.
    """
    input_path = resolve_logo_path(args.input) or args.input
    if not os.path.exists(input_path):
        logger.error(f"[ERROR] Image not found: {input_path}")
        return 0, {}, []

    # Determine output path
    filename = os.path.basename(input_path)
    output_path = os.path.join(args.output, f"{os.path.splitext(filename)[0]}_processed{os.path.splitext(filename)[1]}")

    # Read input image
    try:
        output_image = cv2.imread(input_path)
        if output_image is None:
            logger.error(f"[ERROR] Failed to read image: {input_path}")
            return 0, {}, []
        logger.info(f"[DEBUG] Input image shape: {output_image.shape}")
    except Exception as e:
        logger.error(f"[ERROR] Error reading image: {e}")
        return 0, {}, []

    # Preprocess image
    output_image = preprocess_image(output_image)

    # Initialize vehicle detector
    try:
        detector = VehicleDetector(model_path="yolov8n.pt", use_gpu=args.use_gpu)
    except Exception as e:
        logger.error(f"[ERROR] Failed to initialize VehicleDetector: {e}")
        return 0, {}, []

    # Detect vehicles
    start_time = time.time()
    try:
        vehicle_boxes = detector.detect(
            image=output_image,  # Pass the image array directly
            conf_threshold=args.conf
        )
        vehicle_counts = {"Total": len(vehicle_boxes)}
        for box in vehicle_boxes:
            # Assuming box is a tuple (x1, y1, x2, y2, label, conf)
            label = box[4] if len(box) > 4 else "Vehicle"
            vehicle_counts[label] = vehicle_counts.get(label, 0) + 1
    except Exception as e:
        logger.error(f"[ERROR] Vehicle detection failed: {e}")
        return 0, {}, []

    process_time = time.time() - start_time
    logger.info(f"[SUCCESS] Vehicle detection complete in {process_time:.2f} seconds.")
    logger.info(f"[SUCCESS] Detected {vehicle_counts['Total']} vehicle(s).")

    # Detect license plates
    plate_texts = []
    plate_boxes = []
    if args.detect_plates:
        try:
            plates = extract_license_plates(
                image_path=output_image,
                car_boxes=[(box[0], box[1], box[2], box[3]) for box in vehicle_boxes],
                ocr_engine=args.ocr_engine,
                conf_threshold=args.conf,
                use_gpu=args.use_gpu,
                plate_model_path=args.plate_model,
                region=args.region
            )
            if plates:
                plate_texts = [text for text, _ in plates]
                plate_boxes = [box for _, box in plates]
                logger.info(f"[SUCCESS] Detected {len(plate_texts)} license plates: {plate_texts}")
            else:
                logger.info("[INFO] No license plates detected.")
        except Exception as e:
            logger.error(f"[ERROR] License plate detection failed: {e}")

    # Replace plates with logo if requested
    if args.replace_logo and plate_boxes:
        from vehicle_detection.plate_replacer import PlateReplacer
        replacer = PlateReplacer(use_gpu=args.use_gpu)
        try:
            output_image = replacer.overlay_logo_on_frame(
                frame=output_image,
                plate_boxes=plate_boxes,
                logo=cv2.imread(args.logo, cv2.IMREAD_UNCHANGED) if args.logo else None,
                photorealistic=args.photorealistic
            )
            logger.info(f"[SUCCESS] Successfully replaced {len(plate_boxes)} license plates with logo.")
        except Exception as e:
            logger.error(f"[ERROR] Failed to overlay logo on plates: {e}")

    # Draw vehicle bounding boxes
    colors = generate_random_colors() if args.random_colors else DEFAULT_COLORS
    for box in vehicle_boxes:
        x1, y1, x2, y2 = box[:4]
        label = box[4] if len(box) > 4 else "Vehicle"
        color = colors.get(label, (0, 255, 0))
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, args.thickness)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, 2)[0]
        cv2.rectangle(output_image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
        cv2.putText(output_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, args.font_scale, (255, 255, 255), 2)

    # Save cropped vehicles if requested
    if args.video_crop_dir and vehicle_boxes:
        try:
            os.makedirs(args.video_crop_dir, exist_ok=True)
            for idx, box in enumerate(vehicle_boxes):
                x1, y1, x2, y2 = box[:4]
                if x1 < 0 or y1 < 0 or x2 > output_image.shape[1] or y2 > output_image.shape[0] or x1 >= x2 or y1 >= y2:
                    logger.warning(f"[WARNING] Invalid vehicle coordinates: {(x1, y1, x2, y2)}, skipping")
                    continue
                cropped_vehicle = output_image[y1:y2, x1:x2]
                crop_filename = os.path.join(args.video_crop_dir, f"{os.path.splitext(filename)[0]}_crop_{idx + 1}.jpg")
                cv2.imwrite(crop_filename, cropped_vehicle)
                logger.info(f"[SUCCESS] Cropped vehicle saved to: {crop_filename}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save cropped vehicles: {e}")

    # Log to database
    conn = init_db()
    for idx, box in enumerate(vehicle_boxes):
        vehicle_type = box[4] if len(box) > 4 else "Vehicle"
        plate = plate_texts[idx] if idx < len(plate_texts) else "UNKNOWN"
        confidence = box[5] if len(box) > 5 else 0.9
        log_vehicle(conn, vehicle_type, plate, confidence, box[:4], output_path)
    conn.close()

    # Save detection results to CSV if requested
    if args.save_csv:
        try:
            csv_path = os.path.join(args.output, "detection_summary.csv")
            os.makedirs(args.output, exist_ok=True)
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Image", "Vehicle Count", "Plate Count", "Plate Texts", "Processing Time (s)"])
                writer.writerow([
                    filename,
                    len(vehicle_boxes),
                    len(plate_texts),
                    ", ".join(plate_texts),
                    f"{process_time:.2f}"
                ])
            logger.info(f"[SUCCESS] CSV summary updated at: {csv_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to update CSV summary: {e}")

    # Save detection results to JSON if requested
    if args.save_json:
        try:
            json_path = os.path.join(args.output, "detection_results.json")
            detection_results = {
                "image": filename,
                "vehicle_boxes": [box[:4] for box in vehicle_boxes],  # Only save bbox coordinates
                "vehicle_counts": vehicle_counts,
                "plate_texts": plate_texts,
                "processing_time": process_time
            }
            mode = 'a' if os.path.exists(json_path) else 'w'
            with open(json_path, mode, encoding="utf-8") as json_file:
                json.dump(detection_results, json_file, indent=4)
                json_file.write('\n')
            logger.info(f"[SUCCESS] Detection results saved to: {json_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save detection results to JSON: {e}")

    # Save detection results to XML if requested
    if args.save_xml:
        try:
            xml_path = os.path.join(args.output, "detection_results.xml")
            root = ET.Element("DetectionResults")
            image_elem = ET.SubElement(root, "Image")
            image_elem.set("name", filename)
            image_elem.set("processing_time", f"{process_time:.2f}")

            vehicles_elem = ET.SubElement(image_elem, "Vehicles")
            for idx, box in enumerate(vehicle_boxes):
                vehicle_elem = ET.SubElement(vehicles_elem, "Vehicle")
                vehicle_elem.set("id", str(idx))
                vehicle_elem.set("type", box[4] if len(box) > 4 else "Vehicle")
                vehicle_elem.set("confidence", str(box[5] if len(box) > 5 else 0.9))
                bbox_elem = ET.SubElement(vehicle_elem, "BoundingBox")
                bbox_elem.set("x1", str(box[0]))
                bbox_elem.set("y1", str(box[1]))
                bbox_elem.set("x2", str(box[2]))
                bbox_elem.set("y2", str(box[3]))
                plate_elem = ET.SubElement(vehicle_elem, "LicensePlate")
                plate_elem.text = plate_texts[idx] if idx < len(plate_texts) else "UNKNOWN"

            tree = ET.ElementTree(root)
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            logger.info(f"[SUCCESS] Detection results saved to: {xml_path}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save detection results to XML: {e}")

    # Save output image
    try:
        os.makedirs(args.output, exist_ok=True)
        cv2.imwrite(output_path, output_image)
        logger.info(f"[SUCCESS] Processed image saved to: {output_path}")
    except Exception as e:
        logger.error(f"[ERROR] Failed to save output image: {e}")

    return len(vehicle_boxes), vehicle_counts, plate_texts

def process_batch_file(filepath: str, args) -> Dict:
    """
    Process a single file (image or video) in batch mode.

    Args:
        filepath: Path to the file.
        args: Command-line arguments.

    Returns:
        Dict with processing results.
    """
    filename = os.path.basename(filepath)
    original_input = args.input
    args.input = filepath

    try:
        if any(filepath.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']):
            vehicle_count, counts, plates = process_single_image(args)
            return {"filename": filename, "type": "image", "vehicles": vehicle_count, "counts": counts, "plates": plates}
        elif any(filepath.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov']):
            output_path = os.path.join(args.output, f"{os.path.splitext(filename)[0]}_processed.mp4")
            logo_path = args.logo if args.replace_logo else None
            result = process_video(
                video_path=filepath,
                output_path=output_path,
                logo_path=logo_path,
                detection_interval=15,
                show_progress=True,
                conf_threshold=args.conf,
                use_gpu=args.use_gpu,
                plate_model_path=args.plate_model,
                region=args.region
            )
            if result:
                cap = cv2.VideoCapture(result)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                vehicle_count = frame_count // 30  # Rough estimate
                return {"filename": filename, "type": "video", "vehicles": vehicle_count, "counts": {}, "plates": []}
            else:
                return {"filename": filename, "type": "video", "vehicles": 0, "counts": {}, "plates": []}
    except Exception as e:
        logger.error(f"[ERROR] Error processing {filename}: {e}")
        return {"filename": filename, "type": "unknown", "vehicles": 0, "counts": {}, "plates": []}
    finally:
        args.input = original_input

def process_batch(args):
    """
    Process all images and videos in the input directory using multiprocessing.

    Args:
        args: Command-line arguments.
    """
    input_dir = args.input
    total_vehicles = 0
    total_images = 0
    total_videos = 0
    processed_images = 0
    processed_videos = 0
    all_counts = {"Total": 0}
    all_plates = []

    if not os.path.isdir(input_dir):
        logger.error(f"[ERROR] Input directory not found: {input_dir}")
        return

    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    total_files = len(files)
    logger.info(f"[INFO] Found {total_files} files to process")

    # Use multiprocessing for batch processing
    file_paths = [os.path.join(input_dir, f) for f in files]
    results = []
    for fp in tqdm(file_paths, desc="Processing files", unit="file"):
        result = process_batch_file(fp, args)
        results.append(result)

    # Aggregate results
    for result in results:
        if result["type"] == "image":
            total_images += 1
            processed_images += 1 if result["vehicles"] > 0 else 0
            total_vehicles += result["vehicles"]
            for vehicle_type, count in result["counts"].items():
                if vehicle_type != "Total":
                    all_counts[vehicle_type] = all_counts.get(vehicle_type, 0) + count
            all_counts["Total"] += result["vehicles"]
            all_plates.extend(result["plates"])
        elif result["type"] == "video":
            total_videos += 1
            processed_videos += 1 if result["vehicles"] > 0 else 0
            total_vehicles += result["vehicles"]
            all_counts["Total"] += result["vehicles"]

    # Log summary
    logger.info(
        f"[SUCCESS] Batch processing complete. Processed {processed_images + processed_videos} files "
        f"({processed_images} images, {processed_videos} videos).")
    logger.info(f"[SUCCESS] Detected {total_vehicles} vehicles across all processed files.")
    logger.info(f"[SUCCESS] Detected {len(all_plates)} license plates: {all_plates}")

    # Generate report and plot if requested
    if args.generate_report:
        generate_report(all_counts, all_plates, args.output, process_time=time.time() - start_time)
    if args.plot_results:
        plot_results(all_counts, all_plates, args.output)

def clear_directory(directory: str):
    """
    Clear all files in the specified directory.

    Args:
        directory: Path to the directory to clear.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    logger.debug(f"[DEBUG] Removed file: {file_path}")
            except Exception as e:
                logger.warning(f"[WARNING] Failed to remove file {file_path}: {e}")

def main():
    """
    Main function for the command-line interface.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    global start_time
    start_time = time.time()
    args = parse_arguments()

    # Validate logo path
    args.logo = validate_logo_path(args.logo)

    try:
        os.makedirs(args.output, exist_ok=True)

        # Clear cropped vehicles directory to avoid processing stale files
        clear_directory(args.video_crop_dir)
        clear_directory(args.video_plates_dir)

        # Real-time webcam processing
        if args.webcam:
            logger.info("[INFO] Starting real-time webcam detection...")
            real_time_detection(
                conf_threshold=args.conf,
                camera_id=0,
                save_output=True,
                output_path=args.video_output,
                logo_path=args.logo,
                use_gpu=args.use_gpu,
                plate_model_path=args.plate_model,
                region=args.region
            )
            return 0

        # Process video if requested
        if args.video:
            logger.info(f"[INFO] Processing video: {args.input}")
            result = process_video(
                video_path=args.input,
                output_path=args.video_output,
                logo_path=args.logo,
                detection_interval=15,
                show_progress=True,
                conf_threshold=args.conf,
                use_gpu=args.use_gpu,
                plate_model_path=args.plate_model,
                region=args.region
            )
            if result:
                logger.info(f"[SUCCESS] Video processing completed: {result}")
                return 0
            else:
                logger.error("[ERROR] Video processing failed")
                return 1

        # Process images
        elif os.path.isdir(args.input) or args.batch:
            if not os.path.isdir(args.input):
                logger.error(f"[ERROR] Input must be a directory when using --batch: {args.input}")
                return 1
            logger.info(f"[INFO] Starting batch processing of images and videos in {args.input}")
            process_batch(args)

        else:
            logger.info(f"[INFO] Processing single image: {args.input}")
            vehicle_count, counts, plates = process_single_image(args)
            if args.generate_report:
                generate_report(counts, plates, args.output, process_time=time.time() - start_time)
            if args.plot_results:
                plot_results(counts, plates, args.output)

        # Process detected plates from cropped vehicles
        if args.detect_plates:
            cropped_dir = os.path.join(args.output, "cropped_vehicles")
            plates_dir = args.video_plates_dir
            os.makedirs(plates_dir, exist_ok=True)

            if os.path.exists(cropped_dir):
                cropped_images = [os.path.join(cropped_dir, f) for f in os.listdir(cropped_dir) if f.endswith('.jpg')]
                if not cropped_images:
                    logger.info(f"[INFO] No cropped vehicle images found in: {cropped_dir}")
                    return 0

                # Load EasyOCR once to optimize performance
                from vehicle_detection.plate_recognizer import load_ocr_reader
                ocr_reader = load_ocr_reader(use_gpu=args.use_gpu)

                for img_path in tqdm(cropped_images, desc="Processing cropped vehicles for plates"):
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            logger.warning(f"[WARNING] Could not read image: {img_path}")
                            continue
                        # Assume the entire cropped image is the vehicle
                        height, width = img.shape[:2]
                        car_boxes = [(0, 0, width, height)]
                        plates = extract_license_plates(
                            image_path=img,
                            car_boxes=car_boxes,
                            save_dir=plates_dir,
                            ocr_engine=args.ocr_engine,
                            conf_threshold=args.conf,
                            use_gpu=args.use_gpu,
                            plate_model_path=args.plate_model,
                            region=args.region,
                            ocr_reader=ocr_reader  # Pass the preloaded OCR reader
                        )
                        if plates:
                            for plate_text, _ in plates:
                                logger.info(f"[PLATE] {os.path.basename(img_path)}: {plate_text}")
                        else:
                            logger.warning(f"[WARNING] No plate detected in: {os.path.basename(img_path)}")
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to process image for plate detection: {e}")

        # Print summary dashboard
        print("\n=== Processing Summary ===")
        print(f"Total Processing Time: {time.time() - start_time:.2f} seconds")
        print(f"Output Directory: {args.output}")
        print(f"GPU Used: {args.use_gpu and torch.cuda.is_available()}")
        print(f"Confidence Threshold: {args.conf}")
        print("==========================")
=======
    parser.add_argument("--detect-plates", action="store_true",
                        help="Also detect license plates after vehicle detection")

    return parser.parse_args()


def process_single_image(args) -> Tuple[int, Dict]:
    """
    Process a single image with the given arguments

    Args:
        args: Command-line arguments

    Returns:
        Tuple containing number of detected vehicles and counts by type
    """
    input_path = args.input

    # Determine output path
    filename = os.path.basename(input_path)
    output_path = os.path.join(args.output, f"{os.path.splitext(filename)[0]}_detected{os.path.splitext(filename)[1]}")
    # Read input image for annotations later
    output_image = cv2.imread(input_path)

    # Validate input path
    if not os.path.exists(input_path):
        logger.error(f"[ERROR] Image not found: {input_path}")
        return 0, {}

    # Choose colors
    colors = generate_random_colors() if args.random_colors else DEFAULT_COLORS

    # Process the image
    start_time = time.time()
    boxes, vehicle_counts = detect_vehicles(
        image_path=input_path, 
        save_path=output_path, 
        show=args.show,
        conf_threshold=args.conf,
        custom_colors=colors,
        open_image=args.open,
        line_thickness=args.thickness,
        font_scale=args.font_scale
    )

    process_time = time.time() - start_time
    logger.info(f"[SUCCESS] Detection complete in {process_time:.2f} seconds.")
    logger.info(f"[SUCCESS] Detected {vehicle_counts['Total']} vehicle(s) in {process_time:.2f} seconds.")
    for vehicle_type, count in vehicle_counts.items():
        if vehicle_type != "Total" and count > 0:
            logger.info(f"  - {vehicle_type}: {count}")

    if args.detect_plates:
        # Check if output_image is defined
        if output_image is None:
            logger.error("[ERROR] Cannot perform license plate detection: Output image not available")
            return len(boxes), vehicle_counts

        # Try to import the license plate recognition module
        try:
            from vehicle_detection.plate_recognizer import extract_license_plates, load_ocr_reader

            # Check if OCR reader is available
            ocr_reader = load_ocr_reader()
            if ocr_reader is None:
                logger.error("[ERROR] OCR reader not available. Cannot perform license plate detection.")
                print("[ERROR] OCR reader not available. Make sure easyocr is installed.")
                return len(boxes), vehicle_counts

        except ImportError as e:
            logger.error(f"[ERROR] License plate detection requires easyocr: {e}")
            print("[ERROR] License plate detection requires easyocr. Install with: pip install easyocr")
            return len(boxes), vehicle_counts
        except Exception as e:
            logger.error(f"[ERROR] Failed to import license plate recognition module: {e}")
            print(f"[ERROR] Failed to import license plate recognition module: {e}")
            return len(boxes), vehicle_counts

        try:
            # Prepare bounding boxes for license plate detection
            clean_boxes = []
            try:
                clean_boxes = [
                    tuple(map(int, box.xyxy[0])) for box in boxes
                ]
            except Exception as e:
                logger.error(f"[ERROR] Failed to process bounding boxes for plate detection: {e}")
                return len(boxes), vehicle_counts

            if not clean_boxes:
                logger.info("[INFO] No vehicle boxes available for license plate detection")
                print("[INFO] No vehicle boxes available for license plate detection")
                return len(boxes), vehicle_counts

            # Extract license plates
            logger.info("[INFO] Attempting to detect license plates...")
            plate_data = extract_license_plates(input_path, clean_boxes)

            # Process results
            if plate_data and isinstance(plate_data, list):
                plate_texts = [text for text, _ in plate_data]
                print(f"[SUCCESS] License Plates Detected: {plate_texts}")
                logger.info(f"[SUCCESS] License Plates: {plate_texts}")

                # Create output directories with error handling
                try:
                    os.makedirs("outputs/plates", exist_ok=True)
                except Exception as e:
                    logger.error(f"[ERROR] Failed to create output directory: {e}")
                    print(f"[ERROR] Failed to create output directory: {e}")
                    # Continue processing even if directory creation fails

                # Save text file with error handling
                try:
                    text_file_path = os.path.join("outputs/plates",
                                                f"{os.path.splitext(os.path.basename(input_path))[0]}_plates.txt")
                    with open(text_file_path, "w", encoding="utf-8") as f:
                        for text, _ in plate_data:
                            f.write(f"{text}\n")
                    logger.info(f"[SUCCESS] License plate data saved to {text_file_path}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to save license plate text file: {e}")
                    print(f"[ERROR] Failed to save license plate text file: {e}")
                    # Continue processing even if file saving fails

                # Draw plates on image with error handling
                try:
                    for text, (x1, y1, x2, y2) in plate_data:
                        # Validate coordinates
                        if (x1 < 0 or y1 < 0 or x2 > output_image.shape[1] or y2 > output_image.shape[0] or
                            x1 >= x2 or y1 >= y2):
                            logger.warning(f"[WARNING] Invalid plate coordinates: {(x1, y1, x2, y2)}, skipping")
                            continue

                        # Draw text on image
                        cv2.putText(output_image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                    0.7, (0, 255, 255), 2)
                except Exception as e:
                    logger.error(f"[ERROR] Failed to annotate image with plate text: {e}")
                    print(f"[ERROR] Failed to annotate image with plate text: {e}")
                    # Continue processing even if annotation fails

                # Save updated image with error handling
                try:
                    annotated_path = os.path.join(args.output,
                                                f"{os.path.splitext(os.path.basename(input_path))[0]}_detected.jpg")
                    cv2.imwrite(annotated_path, output_image)
                    logger.info(f"[SUCCESS] Annotated image saved to {annotated_path}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to save annotated image: {e}")
                    print(f"[ERROR] Failed to save annotated image: {e}")
                    # Continue processing even if image saving fails

                # Update CSV summary with error handling
                try:
                    csv_path = "outputs/plate_summary.csv"
                    file_exists = os.path.isfile(csv_path)

                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

                    with open(csv_path, "a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(["Image", "Vehicle Count", "Plate Texts"])
                        writer.writerow([os.path.basename(input_path), len(boxes), ", ".join(plate_texts)])
                    logger.info(f"[SUCCESS] CSV summary updated at {csv_path}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to update CSV summary: {e}")
                    print(f"[ERROR] Failed to update CSV summary: {e}")
                    # Continue processing even if CSV update fails
            else:
                print("[INFO] No license plates detected.")
                logger.info("[INFO] No license plates detected.")

        except Exception as e:
            logger.error(f"[ERROR] License plate detection failed: {e}")
            print(f"[ERROR] License plate detection failed: {e}")
            # Continue with the rest of the processing

    # Print results
    if vehicle_counts.get("Total", 0) > 0:
        logger.info(f"[SUCCESS] Detected {vehicle_counts['Total']} vehicle(s) in {process_time:.2f} seconds.")
        for vehicle_type, count in vehicle_counts.items():
            if vehicle_type != "Total" and count > 0:
                logger.info(f"  - {vehicle_type}: {count}")
        logger.info(f"[SUCCESS] Output saved to: {output_path}")
    else:
        logger.info(f"[INFO] No vehicles detected in {process_time:.2f} seconds.")

    return len(boxes), vehicle_counts
    # If license plate detection is enabled

def main():
    """
    Main function for the command-line interface

    Returns:
        Exit code (0 for success, 1 for error)
    """
    # Parse command line arguments
    args = parse_arguments()

    try:
        # Create output directory if it doesn't exist
        os.makedirs(args.output, exist_ok=True)

        # Process images
        if args.batch:
            if not os.path.isdir(args.input):
                logger.error(f"[ERROR] Input must be a directory when using --batch: {args.input}")
                return 1

            logger.info(f"[INFO] Starting batch processing of images in {args.input}")
            start_time = time.time()

            # Choose colors
            colors = generate_random_colors() if args.random_colors else DEFAULT_COLORS

            # Process all images in the directory
            results = batch_process(
                input_dir=args.input,
                output_dir=args.output,
                conf_threshold=args.conf,
                custom_colors=colors,
                show=args.show,
                open_images=args.open
            )

            process_time = time.time() - start_time

            # Calculate total vehicles
            total_vehicles = sum(counts.get("Total", 0) for counts in results.values())
            logger.info(f"[SUCCESS] Batch processing complete in {process_time:.2f} seconds.")
            logger.info(f"[SUCCESS] Processed {len(results)} images, detected {total_vehicles} vehicles total.")

        else:
            # Process a single image
            process_single_image(args)
>>>>>>> origin/main

    except KeyboardInterrupt:
        logger.info("[WARNING] Processing interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"[ERROR] An error occurred: {str(e)}")
        return 1

    return 0

<<<<<<< HEAD
def main_cli():
    """
    Entry point for the command-line interface.
    """
    sys.exit(main())

if __name__ == "__main__":
    main_cli()
=======

def main_cli():
    """
    Entry point for the command-line interface
    """
    sys.exit(main())


if __name__ == "__main__":
    main_cli()
>>>>>>> origin/main
