"""
Vehicle Detection CLI
====================

This module provides a command-line interface for the vehicle detection system.

It includes:
- Command-line argument parsing
- Single image processing
- Batch processing of directories

- Error handling and logging

"""

import sys
import csv

import cv2
import os
import argparse
import logging
import time
from typing import Dict, Tuple


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


    # Visualization options
    parser.add_argument("--random-colors", action="store_true",
                        help="Use random colors for different vehicle types")
    parser.add_argument("--thickness", type=int, default=2,
                        help="Line thickness for bounding boxes")
    parser.add_argument("--font-scale", type=float, default=0.6,
                        help="Font scale for labels")

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


    except KeyboardInterrupt:
        logger.info("[WARNING] Processing interrupted by user.")
        return 1
    except Exception as e:
        logger.error(f"[ERROR] An error occurred: {str(e)}")
        return 1

    return 0



def main_cli():
    """
    Entry point for the command-line interface
    """
    sys.exit(main())


if __name__ == "__main__":
    main_cli()

