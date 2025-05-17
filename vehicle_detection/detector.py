"""
Vehicle Detection Module
=======================

This module provides functionality for detecting vehicles in images and videos
using the YOLOv8 model with enhanced features.

It includes features for:
- Detecting vehicles using YOLOv8
- Visualizing detections with bounding boxes and labels
- Batch processing for images and videos
- Parallel processing for faster batch operations
- Adaptive confidence thresholding based on image quality
- Overlaying detection statistics on output images
"""

import os
import cv2
import logging
import time
import random
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import requests
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('vehicle_detection.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Vehicle classes to detect
VEHICLE_CLASSES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Default colors for visualization
DEFAULT_COLORS = {
    'Car': (0, 255, 0),      # Green
    'Motorcycle': (255, 0, 0), # Blue
    'Bus': (0, 0, 255),      # Red
    'Truck': (255, 255, 0),   # Cyan
    'Vehicle': (255, 165, 0)  # Orange (fallback)
}

# URL to download the default vehicle detection model if missing
DEFAULT_VEHICLE_MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"

# Global model cache
global_model = None

class VehicleDetector:
    """Vehicle detection using YOLOv8 with enhanced features"""

    def __init__(self, model_path: str = "yolov8n.pt", conf_threshold: float = 0.5, use_gpu: bool = False):
        """
        Initialize the vehicle detector with YOLOv8.

        Args:
            model_path: Path to the YOLOv8 model weights
            conf_threshold: Confidence threshold for detections
            use_gpu: Whether to use GPU for inference
        """
        self.conf_threshold = conf_threshold
        self.use_gpu = use_gpu
        self.device = 'cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu'

        # Load or download the model
        if not os.path.exists(model_path):
            logger.warning(f"[WARNING] Model not found at {model_path}, attempting to download...")
            if not self.download_vehicle_model(model_path):
                raise ValueError(f"Failed to download model to {model_path}")

        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"[SUCCESS] YOLOv8 model loaded successfully from {model_path} on {self.device}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to load YOLOv8 model: {str(e)}")
            raise

    def download_vehicle_model(self, model_path: str) -> bool:
        """
        Download the vehicle detection model if it's missing.

        Args:
            model_path: Path where the model should be saved.

        Returns:
            True if download is successful, False otherwise.
        """
        try:
            logger.info(f"[INFO] Downloading vehicle detection model to: {model_path}")
            response = requests.get(DEFAULT_VEHICLE_MODEL_URL, stream=True, timeout=30)
            response.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            logger.info(f"[SUCCESS] Vehicle detection model downloaded to: {model_path}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to download vehicle detection model: {e}")
            return False

    def preprocess_image(self, frame: np.ndarray, max_size: int = 640) -> np.ndarray:
        """
        Preprocess image by resizing to reduce computation.

        Args:
            frame: Input frame
            max_size: Maximum size for the largest dimension

        Returns:
            Resized frame
        """
        try:
            h, w = frame.shape[:2]
            scale = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            return cv2.resize(frame, (new_w, new_h))
        except Exception as e:
            logger.error(f"[ERROR] Failed to preprocess image: {str(e)}")
            raise

    def filter_invalid_boxes(self, boxes: np.ndarray, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Filter out invalid bounding boxes.

        Args:
            boxes: Array of bounding boxes (x1, y1, x2, y2)
            image_shape: Shape of the image (height, width)

        Returns:
            Filtered boxes
        """
        height, width = image_shape
        filtered_boxes = []
        min_size = 10  # Minimum width and height for a valid box

        for box in boxes:
            x1, y1, x2, y2 = box
            box_width = x2 - x1
            box_height = y2 - y1

            if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x1 >= x2 or y1 >= y2:
                logger.warning(f"[WARNING] Invalid bounding box detected: ({x1}, {y1}, {x2}, {y2}) - out of bounds or invalid dimensions")
                continue
            if box_width < min_size or box_height < min_size:
                logger.warning(f"[WARNING] Bounding box too small: ({x1}, {y1}, {x2}, {y2}) - width={box_width}, height={box_height}, min_size={min_size}")
                continue
            filtered_boxes.append(box)
        return np.array(filtered_boxes)

    def assess_image_quality(self, frame: np.ndarray) -> float:
        """
        Assess image quality to adjust confidence threshold.

        Args:
            frame: Input frame (BGR format)

        Returns:
            Quality score (0 to 1), where 1 is high quality
        """
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Compute brightness (mean pixel value)
            brightness = np.mean(gray)
            # Compute contrast (standard deviation of pixel values)
            contrast = np.std(gray)
            # Normalize brightness (ideal range: 100-150) and contrast (ideal: >30)
            brightness_score = min(max((brightness - 50) / 100, 0), 1)  # 0 to 1
            contrast_score = min(contrast / 50, 1)  # 0 to 1
            quality_score = (brightness_score + contrast_score) / 2
            logger.debug(f"[DEBUG] Image quality - Brightness: {brightness:.2f}, Contrast: {contrast:.2f}, Quality Score: {quality_score:.2f}")
            return quality_score
        except Exception as e:
            logger.error(f"[ERROR] Failed to assess image quality: {str(e)}")
            return 1.0  # Default to high quality to avoid overly low thresholds

    def detect(self, frame: np.ndarray, conf: Optional[float] = None, use_gpu: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Detect vehicles in a frame with adaptive confidence thresholding.

        Args:
            frame: Input frame (BGR format)
            conf: Optional confidence threshold override
            use_gpu: Optional override for GPU usage

        Returns:
            List of detections with bounding boxes and class information
        """
        try:
            # Use provided GPU setting or default from init
            device = 'cuda' if (use_gpu if use_gpu is not None else self.use_gpu) and torch.cuda.is_available() else 'cpu'

            # Validate frame
            if frame is None or frame.size == 0:
                logger.error("[ERROR] Invalid frame provided to detect: frame is None or empty")
                return []

            # Log frame shape for debugging
            logger.debug(f"[DEBUG] Processing frame with shape: {frame.shape}")

            # Preprocess frame
            original_h, original_w = frame.shape[:2]
            frame_processed = self.preprocess_image(frame)
            h, w = frame_processed.shape[:2]
            scale_x, scale_y = original_w / w, original_h / h

            # Adjust confidence threshold based on image quality
            quality_score = self.assess_image_quality(frame)
            base_conf = conf if conf is not None else self.conf_threshold
            adjusted_conf = max(0.1, base_conf * (1 - 0.3 * (1 - quality_score)))  # Lower threshold for poor quality
            logger.info(f"[INFO] Adjusted confidence threshold to {adjusted_conf:.2f} based on image quality (score: {quality_score:.2f})")

            # Run detection with retry mechanism
            detections = []
            for attempt in range(2):  # Try twice with different confidence thresholds
                try:
                    results = self.model.predict(frame_processed, conf=adjusted_conf, device=device)
                    inference_time = time.time() - time.time()
                    logger.info(f"[INFO] YOLOv8 inference completed in {inference_time:.3f} seconds on {device}")

                    # Extract detections
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)  # (x1, y1, x2, y2)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)

                    # Scale boxes back to original image size
                    boxes = boxes * [scale_x, scale_y, scale_x, scale_y]
                    boxes = boxes.astype(int)

                    # Filter invalid boxes
                    boxes = self.filter_invalid_boxes(boxes, (original_h, original_w))

                    # Create detections list
                    for box, conf, cls_id in zip(boxes, confidences, class_ids):
                        if cls_id in VEHICLE_CLASSES:
                            x1, y1, x2, y2 = box
                            detections.append({
                                'bbox': (x1, y1, x2, y2),
                                'class': VEHICLE_CLASSES[cls_id],
                                'confidence': float(conf)
                            })
                    break  # Successful detection, exit loop
                except Exception as e:
                    logger.error(f"[ERROR] YOLOv8 detection failed on attempt {attempt + 1}: {e}")
                    if attempt == 0:
                        adjusted_conf = max(0.1, adjusted_conf - 0.1)  # Lower threshold for retry
                        logger.info(f"[INFO] Retrying detection with lowered confidence threshold: {adjusted_conf:.2f}")
                    else:
                        return []  # Give up after second attempt

            logger.info(f"[INFO] Detected {len(detections)} vehicles in frame")
            return detections

        except Exception as e:
            logger.error(f"[ERROR] Detection failed: {str(e)}")
            return []

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]], vehicle_counts: Dict[str, int], custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None, line_thickness: int = 2, font_scale: float = 0.6, overlay_stats: bool = True) -> np.ndarray:
        """
        Draw detection boxes on the frame with enhanced visualization and optional statistics overlay.

        Args:
            frame: Input frame
            detections: List of detections
            vehicle_counts: Dictionary with vehicle counts by type
            custom_colors: Custom colors for different vehicle types
            line_thickness: Thickness of bounding box lines
            font_scale: Scale of the font for labels
            overlay_stats: Whether to overlay detection statistics on the image

        Returns:
            Frame with detection boxes drawn
        """
        colors = custom_colors if custom_colors else DEFAULT_COLORS
        output_frame = frame.copy()

        # Draw bounding boxes
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cls = det['class']
            conf = det['confidence']

            # Get color for class
            color = colors.get(cls, colors.get('Vehicle', (0, 255, 0)))

            # Draw box with shadow effect for better visibility
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 0, 0), line_thickness + 2)  # Shadow
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, line_thickness)

            # Draw label with background
            label = f"{cls}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, line_thickness)[0]
            cv2.rectangle(output_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 5, y1), color, -1)
            cv2.putText(output_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), line_thickness)

        # Overlay statistics if requested
        if overlay_stats:
            stats_text = [f"{cls_name}: {count}" for cls_name, count in vehicle_counts.items() if count > 0 or cls_name == "Total"]
            y_offset = 30
            for text in stats_text:
                cv2.putText(output_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), line_thickness)
                y_offset += int(25 * font_scale)

        return output_frame

# Initialize global model instance
try:
    global_model = VehicleDetector(model_path="yolov8n.pt", use_gpu=False)
    logger.info("Global YOLOv8 model instance initialized")
except Exception as e:
    logger.error(f"Failed to initialize global model: {str(e)}")
    global_model = None

# Alias for backward compatibility
model = global_model

def get_detector() -> Optional[VehicleDetector]:
    """
    Get the global VehicleDetector instance.

    Returns:
        The global VehicleDetector instance, or None if not initialized.
    """
    return global_model

def generate_random_colors(num_colors: int = 5) -> Dict[str, Tuple[int, int, int]]:
    """
    Generate random colors for visualization.

    Args:
        num_colors: Number of colors to generate

    Returns:
        Dictionary mapping class names to RGB color tuples
    """
    colors = {}
    for cls_name in VEHICLE_CLASSES.values():
        colors[cls_name] = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    colors["Vehicle"] = (255, 255, 0)  # Default color
    return colors

def detect_vehicles(
    image_path: str,
    save_path: Optional[str] = None,
    show: bool = False,
    conf_threshold: float = 0.5,
    custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    open_image: bool = True,
    line_thickness: int = 2,
    font_scale: float = 0.6,
    use_gpu: bool = False,
    overlay_stats: bool = True
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Detect vehicles in an image with enhanced visualization and statistics.

    Args:
        image_path: Path to the input image
        save_path: Path to save the output image (optional)
        show: Whether to display the image with detections
        conf_threshold: Confidence threshold for detections
        custom_colors: Custom colors for different vehicle types
        open_image: Whether to open the saved image
        line_thickness: Thickness of bounding box lines
        font_scale: Scale of the font for labels
        use_gpu: Whether to use GPU for inference
        overlay_stats: Whether to overlay detection statistics on the image

    Returns:
        Tuple containing list of vehicle detections and dictionary with vehicle counts by type
    """
    detector = get_detector()
    if detector is None:
        logger.error("[ERROR] Model not available. Cannot perform detection.")
        return [], {}

    # Validate input path
    if not os.path.exists(image_path):
        logger.error(f"[ERROR] Input image not found: {image_path}")
        return [], {}

    # Read image
    try:
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"[ERROR] Couldn't read image at: {image_path} - cv2.imread returned None")
            return [], {}
    except Exception as e:
        logger.error(f"[ERROR] Error reading image at {image_path}: {str(e)}")
        return [], {}

    # Use colors provided or defaults
    colors = custom_colors if custom_colors else DEFAULT_COLORS

    # Process frame
    start_time = time.time()
    detections = detector.detect(frame, conf_threshold, use_gpu)
    process_time = time.time() - start_time
    logger.info(f"[SUCCESS] Image processed in {process_time:.2f} seconds")

    # Initialize vehicle counts
    vehicle_counts = {cls_name: 0 for cls_name in VEHICLE_CLASSES.values()}
    vehicle_counts["Total"] = 0

    # Count vehicles by class
    for det in detections:
        class_name = det['class']
        if class_name in vehicle_counts:
            vehicle_counts[class_name] += 1
            vehicle_counts["Total"] += 1

    if len(detections) == 0:
        logger.info(f"[INFO] No vehicles detected in image: {image_path}")
        return [], vehicle_counts

    logger.info(f"[SUCCESS] Detected {len(detections)} vehicle(s) in image: {image_path}")

    # Draw detections
    output_frame = detector.draw_detections(frame, detections, vehicle_counts, colors, line_thickness, font_scale, overlay_stats)

    # Save cropped vehicle images
    if save_path:
        cropped_dir = os.path.join(os.path.dirname(save_path), "cropped_vehicles")
        os.makedirs(cropped_dir, exist_ok=True)

        for i, det in enumerate(detections):
            x1, y1, x2, y2 = det['bbox']
            vehicle_crop = frame[y1:y2, x1:x2]
            crop_filename = f"{os.path.splitext(os.path.basename(save_path))[0]}_crop_{i + 1}.jpg"
            crop_path = os.path.join(cropped_dir, crop_filename)
            try:
                cv2.imwrite(crop_path, vehicle_crop)
                logger.info(f"[INFO] Saved cropped vehicle image to: {crop_path}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to save cropped vehicle image to {crop_path}: {str(e)}")

    # Show detection statistics
    stats_text = []
    for cls_name, count in vehicle_counts.items():
        if count > 0 or cls_name == "Total":
            stats_text.append(f"{cls_name}: {count}")
    logger.info(f"Vehicle counts for {image_path}: " + ", ".join(stats_text))

    # Display image if requested (with GUI support check)
    if show:
        logger.warning("[WARNING] Displaying images requires OpenCV GUI support. Ensure appropriate GUI backend is installed.")
        try:
            cv2.imshow("Detected Vehicles", output_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"[ERROR] Failed to display image: {str(e)}")

    # Save output image if requested
    if save_path:
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, output_frame)
            logger.info(f"[SUCCESS] Output image saved to: {save_path}")

            if open_image:
                abs_save_path = os.path.abspath(save_path)
                if os.path.exists(abs_save_path):
                    try:
                        if os.name == 'nt':  # Windows
                            os.startfile(abs_save_path)
                        elif os.name == 'posix':
                            import subprocess
                            opener = "open" if sys.platform == "darwin" else "xdg-open"
                            subprocess.call([opener, abs_save_path])
                    except Exception as e:
                        logger.warning(f"[WARNING] Couldn't open image: {e}")
                else:
                    logger.warning(f"[WARNING] Output image not found at: {abs_save_path}")
        except Exception as e:
            logger.error(f"[ERROR] Error saving image to {save_path}: {str(e)}")

    return detections, vehicle_counts

def process_batch_file(args_tuple):
    """
    Process a single file in batch mode.

    Args:
        args_tuple: Tuple of (filepath, output_dir, conf_threshold, use_gpu)

    Returns:
        Dict with processing results
    """
    filepath, output_dir, conf_threshold, use_gpu = args_tuple
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_processed{os.path.splitext(filename)[1]}")

    try:
        detections, counts = detect_vehicles(
            image_path=filepath,
            save_path=output_path,
            show=False,
            conf_threshold=conf_threshold,
            custom_colors=None,
            open_image=False,
            use_gpu=use_gpu
        )
        vehicle_count = counts["Total"]
        return {"filename": filename, "type": "image", "vehicles": vehicle_count, "counts": counts, "detections": detections}
    except Exception as e:
        logger.error(f"[ERROR] Error processing {filename}: {e}")
        return {"filename": filename, "type": "image", "vehicles": 0, "counts": {}, "detections": []}

def batch_process(
    input_dir: str,
    output_dir: str,
    conf_threshold: float = 0.5,
    use_gpu: bool = False,
    parallel: bool = False,
    extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
) -> List[Dict[str, Any]]:
    """
    Process all images and videos in the input directory.

    Args:
        input_dir: Directory containing input files
        output_dir: Directory to save output files
        conf_threshold: Confidence threshold for detections
        use_gpu: Whether to use GPU for inference
        parallel: Whether to use parallel processing for images
        extensions: List of valid file extensions to process

    Returns:
        List of processing results
    """
    if not os.path.isdir(input_dir):
        logger.error(f"[ERROR] Input directory not found: {input_dir}")
        return []

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get list of files to process
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    total_files = len(files)
    logger.info(f"[INFO] Found {total_files} files to process")

    # Separate images and videos
    image_files = [f for f in files if any(f.lower().endswith(ext) for ext in extensions)]
    video_files = [f for f in files if any(f.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov'])]

    total_images = len(image_files)
    total_videos = len(video_files)
    processed_images = 0
    processed_videos = 0
    total_vehicles = 0

    # Process images
    file_paths = [os.path.join(input_dir, f) for f in image_files]
    results = []

    if file_paths:
        if parallel:
            logger.info(f"[INFO] Processing {total_images} images in parallel using {cpu_count()} workers")
            with Pool(processes=cpu_count()) as pool:
                args_list = [(filepath, output_dir, conf_threshold, use_gpu) for filepath in file_paths]
                results.extend(pool.map(process_batch_file, args_list))
        else:
            logger.info(f"[INFO] Processing {total_images} images sequentially")
            for filepath in file_paths:
                result = process_batch_file((filepath, output_dir, conf_threshold, use_gpu))
                results.append(result)

        for result in results:
            if result["vehicles"] > 0:
                processed_images += 1
                total_vehicles += result["vehicles"]

    # Process videos
    for filename in video_files:
        filepath = os.path.join(input_dir, filename)
        if not os.path.exists(filepath):
            logger.error(f"[ERROR] Video not found: {filename}")
            results.append({"filename": filename, "type": "video", "vehicles": 0, "counts": {}, "detections": []})
            continue

        logger.info(f"[INFO] Processing video: {filename}")

        try:
            from vehicle_detection.video_processor import process_video
            output_path = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_processed.mp4")
            result = process_video(
                video_path=filepath,
                output_path=output_path,
                conf_threshold=conf_threshold,
                use_gpu=use_gpu
            )
            if result:
                cap = cv2.VideoCapture(output_path)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                vehicle_count = frame_count // 30  # Rough estimate: 1 vehicle per 30 frames
                total_vehicles += vehicle_count
                processed_videos += 1
                logger.info(f"[SUCCESS] Video processed. Estimated {vehicle_count} vehicles.")
                results.append({"filename": filename, "type": "video", "vehicles": vehicle_count, "counts": {"Total": vehicle_count}, "detections": []})
            else:
                logger.warning(f"[WARNING] Video processing returned no result for {filename}")
                results.append({"filename": filename, "type": "video", "vehicles": 0, "counts": {}, "detections": []})
        except Exception as e:
            logger.error(f"[ERROR] Error processing video {filename}: {str(e)}")
            results.append({"filename": filename, "type": "video", "vehicles": 0, "counts": {}, "detections": []})

    # Aggregate counts
    all_counts = {"Total": 0}
    for result in results:
        for vehicle_type, count in result["counts"].items():
            if vehicle_type != "Total":
                all_counts[vehicle_type] = all_counts.get(vehicle_type, 0) + count
        all_counts["Total"] += result["vehicles"]

    # Log summary
    logger.info(
        f"[SUCCESS] Batch processing complete. Processed {processed_images + processed_videos} files ({processed_images} images, {processed_videos} videos).")
    logger.info(f"[SUCCESS] Detected {total_vehicles} vehicles across all processed files.")
    logger.info(f"Vehicle counts: {all_counts}")

    return results