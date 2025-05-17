"""
License Plate Recognition Module
===============================

This module provides functionality for detecting and recognizing license plates in images
using OpenCV, YOLOv8 for plate detection, and OCR techniques (EasyOCR or Tesseract).

It includes features for:
- Detecting license plates within vehicle bounding boxes using YOLOv8
- Recognizing plate text with EasyOCR or Tesseract
- Validating plate text against regional formats
- Saving cropped license plate regions
- Batch processing for efficiency
- Confidence scoring for plate detection and OCR
"""

import os
import cv2
import logging
import numpy as np
import re
import torch
import requests
from typing import List, Tuple, Optional, Dict, Any, Union
from ultralytics import YOLO

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

# Check for EasyOCR and pytesseract availability
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("[WARNING] EasyOCR not available. Install it using 'pip install easyocr'")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("[WARNING] pytesseract not available. Install it using 'pip install pytesseract'")

# Define regional plate formats (simplified examples)
PLATE_FORMATS = {
    "India": r"^[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}$",  # e.g., MH12AB1234
    "US": r"^[A-Z0-9]{3}-?[A-Z0-9]{3,4}$",     # e.g., ABC-123 or ABC1234
    "EU": r"^[A-Z]{1,3}-?\d{1,4}[A-Z]{0,2}$"   # e.g., ABC-123 or AB123CD
}

# URL to download the default plate detection model if missing
DEFAULT_PLATE_MODEL_URL = "https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8/raw/main/license_plate_detector.pt"

# Global cache for OCR reader and plate model
global_ocr_reader = None
global_plate_model = None

def validate_plate_text(text: str, region: str = "India") -> bool:
    """
    Validate the license plate text against regional formats.

    Args:
        text: The extracted plate text
        region: The region to validate against (e.g., "India", "US", "EU")

    Returns:
        True if the text matches the regional format, False otherwise
    """
    if not text or text == "UNKNOWN":
        return False

    if region not in PLATE_FORMATS:
        logger.warning(f"[WARNING] Unsupported region for plate validation: {region}")
        return True  # Skip validation for unsupported regions

    pattern = PLATE_FORMATS[region]
    text = text.strip().replace(" ", "").upper()
    if re.match(pattern, text):
        logger.info(f"[INFO] Plate text '{text}' matches {region} format")
        return True
    else:
        logger.debug(f"[DEBUG] Plate text '{text}' does not match {region} format")
        return False

def download_plate_model(model_path: str) -> bool:
    """
    Download the plate detection model if it's missing and rename it to match the expected filename.

    Args:
        model_path: Path where the model should be saved (expected to be 'yolov8_plate.pt').

    Returns:
        True if download and renaming are successful, False otherwise.
    """
    try:
        # Define a temporary path for the downloaded file
        temp_model_path = os.path.join(os.path.dirname(model_path), "license_plate_detector.pt")

        # Download the model
        logger.info(f"[INFO] Downloading plate detection model to: {temp_model_path}")
        response = requests.get(DEFAULT_PLATE_MODEL_URL, stream=True, timeout=30)
        response.raise_for_status()
        with open(temp_model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        logger.info(f"[SUCCESS] Plate detection model downloaded to: {temp_model_path}")

        # Rename the downloaded file to match the expected model_path (yolov8_plate.pt)
        if temp_model_path != model_path:
            os.rename(temp_model_path, model_path)
            logger.info(f"[INFO] Renamed downloaded model to: {model_path}")

        return True
    except Exception as e:
        logger.error(f"[ERROR] Failed to download or rename plate detection model: {e}")
        # Clean up if the download partially succeeded but renaming failed
        if os.path.exists(temp_model_path):
            try:
                os.remove(temp_model_path)
                logger.info(f"[INFO] Cleaned up temporary file: {temp_model_path}")
            except Exception as cleanup_error:
                logger.error(f"[ERROR] Failed to clean up temporary file: {cleanup_error}")
        return False

def load_ocr_reader(ocr_engine: str = "easyocr", use_gpu: bool = False, custom_ocr_model_path: Optional[str] = None) -> Optional[Any]:
    """
    Load the OCR reader based on the specified engine.

    Args:
        ocr_engine: The OCR engine to use ("easyocr" or "tesseract")
        use_gpu: Whether to use GPU for OCR (applies to EasyOCR)
        custom_ocr_model_path: Path to a custom OCR model (applies to EasyOCR)

    Returns:
        OCR reader instance or None if unavailable
    """
    global global_ocr_reader
    if global_ocr_reader is not None:
        logger.info("[INFO] Using cached OCR reader")
        return global_ocr_reader

    if ocr_engine == "easyocr":
        if not EASYOCR_AVAILABLE:
            logger.error("[ERROR] EasyOCR is not installed. Please install it using 'pip install easyocr'.")
            return None
        try:
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
            if custom_ocr_model_path:
                global_ocr_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'), model_storage_directory=custom_ocr_model_path)
                logger.info(f"[SUCCESS] EasyOCR reader loaded with custom model from {custom_ocr_model_path} on {device}")
            else:
                global_ocr_reader = easyocr.Reader(['en'], gpu=(device == 'cuda'))
                logger.info(f"[SUCCESS] EasyOCR reader loaded (GPU: {device == 'cuda'})")
            return global_ocr_reader
        except Exception as e:
            logger.error(f"[ERROR] Failed to load EasyOCR reader: {e}")
            return None
    elif ocr_engine == "tesseract":
        if not TESSERACT_AVAILABLE:
            logger.error("[ERROR] pytesseract is not installed. Please install it using 'pip install pytesseract'.")
            return None
        try:
            pytesseract.get_tesseract_version()  # Verify Tesseract installation
            global_ocr_reader = pytesseract
            logger.info("[SUCCESS] Tesseract OCR loaded successfully")
            return global_ocr_reader
        except Exception as e:
            logger.error(f"[ERROR] Failed to load Tesseract OCR: {e}")
            return None
    else:
        logger.error(f"[ERROR] Unsupported OCR engine: {ocr_engine}")
        return None

def preprocess_image(image: np.ndarray, max_size: int = 640) -> Optional[np.ndarray]:
    """
    Preprocess the image by resizing to reduce computation while preserving aspect ratio.

    Args:
        image: Input image as numpy array
        max_size: Maximum size for the largest dimension

    Returns:
        Preprocessed image
    """
    try:
        h, w = image.shape[:2]
        scale = min(max_size / h, max_size / w)
        new_h, new_w = int(h * scale), int(w * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception as e:
        logger.error(f"[ERROR] Failed to preprocess image: {e}")
        return image

def preprocess_image_for_ocr(image: np.ndarray, retry: bool = False) -> Optional[np.ndarray]:
    """
    Preprocess the image specifically for OCR to enhance text readability.

    Args:
        image: Input image (BGR format)
        retry: If True, use alternative preprocessing for retry attempt

    Returns:
        Preprocessed image or None if preprocessing fails
    """
    if image is None or image.size == 0:
        logger.error("[ERROR] Empty image provided for OCR preprocessing")
        return None
    if len(image.shape) < 2 or image.shape[0] == 0 or image.shape[1] == 0:
        logger.error("[ERROR] Invalid image dimensions provided for OCR preprocessing")
        return None

    try:
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

        # Enhance contrast using histogram equalization
        gray = cv2.equalizeHist(gray)

        if retry:
            # Alternative preprocessing for retry: Sharpen the image
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            gray = cv2.filter2D(gray, -1, kernel)
            logger.debug("[DEBUG] Applied sharpening filter for OCR retry")
        else:
            # Apply Gaussian blur to reduce noise
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Adaptive thresholding to binarize the image
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        return thresh
    except Exception as e:
        logger.error(f"[ERROR] Failed to preprocess image for OCR: {e}")
        return None

def detect_plates_with_contours(image: np.ndarray, vehicle_box: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
    """
    Detect license plate regions using contour detection as a fallback.

    Args:
        image: Input image (BGR format)
        vehicle_box: Vehicle bounding box (x1, y1, x2, y2)

    Returns:
        List of detected plate bounding boxes (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = vehicle_box
    vehicle_crop = image[y1:y2, x1:x2]
    if vehicle_crop.size == 0:
        logger.warning("[WARNING] Empty vehicle crop for contour detection")
        return []

    try:
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY) if len(vehicle_crop.shape) == 3 else vehicle_crop
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plate_boxes = []

        for contour in contours:
            # Approximate the contour to a polygon
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # Check if the contour is a rectangle (4 sides) and has a plate-like aspect ratio
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / max(1, h)
                area = w * h
                # Simplified filters for typical plate dimensions
                if 2 < aspect_ratio < 5 and 500 < area < 5000 and w > 50 and h > 10:
                    plate_x1 = x1 + x
                    plate_y1 = y1 + y
                    plate_x2 = plate_x1 + w
                    plate_y2 = plate_y1 + h
                    plate_boxes.append((plate_x1, plate_y1, plate_x2, plate_y2))

        logger.info(f"[INFO] Detected {len(plate_boxes)} potential plate regions using contours")
        return plate_boxes
    except Exception as e:
        logger.error(f"[ERROR] Contour-based plate detection failed: {e}")
        return []

def extract_license_plates(
    image_path: Union[str, np.ndarray],
    car_boxes: List[Tuple[int, int, int, int]],
    save_dir: Optional[str] = "outputs/plates/cropped",
    ocr_engine: str = "easyocr",
    conf_threshold: float = 0.3,
    use_gpu: bool = False,
    custom_ocr_model_path: Optional[str] = None,
    plate_model_path: str = "yolov8_plate.pt",
    region: str = "India",
    ocr_reader: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Extract license plates from the given image within vehicle bounding boxes.

    Args:
        image_path: Path to the input image or image array (BGR format)
        car_boxes: List of vehicle bounding boxes (x1, y1, x2, y2)
        save_dir: Directory to save detected plates (optional)
        ocr_engine: OCR engine to use ("easyocr" or "tesseract")
        conf_threshold: Confidence threshold for OCR and plate detection
        use_gpu: Whether to use GPU for OCR and plate detection
        custom_ocr_model_path: Path to a custom OCR model (applies to EasyOCR)
        plate_model_path: Path to the fine-tuned YOLOv8 model for plate detection
        region: Region for plate text validation (e.g., "India", "US", "EU")
        ocr_reader: Preloaded OCR reader to avoid reloading (optional)

    Returns:
        List of dictionaries containing plate text, bounding box, plate confidence, and OCR confidence
    """
    logger.info(f"[INFO] Extracting license plates from {len(car_boxes)} vehicle boxes")
    logger.info(f"[INFO] Using OCR engine: {ocr_engine}, Confidence threshold: {conf_threshold}, Region: {region}")

    # Load image if a path is provided
    if isinstance(image_path, str):
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"[ERROR] Failed to load image: {image_path}")
            return []
    else:
        img = image_path

    # Validate image
    if img is None or img.size == 0:
        logger.error("[ERROR] Invalid image provided for plate extraction")
        return []

    # Preprocess image for efficiency
    img = preprocess_image(img)
    img_height, img_width = img.shape[:2]
    logger.debug(f"[DEBUG] Image dimensions: {img_width}x{img_height}")

    if not car_boxes:
        logger.warning("[WARNING] No vehicle boxes provided for plate extraction")
        return []

    # Load plate detection model
    global global_plate_model
    if global_plate_model is not None:
        plate_detector = global_plate_model
        logger.info("[INFO] Using cached plate detection model")
    else:
        if os.path.exists(plate_model_path):
            try:
                device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                global_plate_model = YOLO(plate_model_path)
                global_plate_model.to(device)
                plate_detector = global_plate_model
                logger.info(f"[SUCCESS] Loaded plate detection model from {plate_model_path} on {device}")
            except Exception as e:
                logger.error(f"[ERROR] Failed to load YOLOv8 plate detector: {e}")
                plate_detector = None
        else:
            logger.warning(f"[WARNING] Plate model not found at {plate_model_path}, attempting to download...")
            if download_plate_model(plate_model_path):
                try:
                    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
                    global_plate_model = YOLO(plate_model_path)
                    global_plate_model.to(device)
                    plate_detector = global_plate_model
                    logger.info(f"[SUCCESS] Loaded downloaded plate detection model on {device}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to load downloaded YOLOv8 plate detector: {e}")
                    plate_detector = None
            else:
                plate_detector = None

    # Load OCR reader if not provided
    if ocr_reader is None:
        ocr_reader = load_ocr_reader(ocr_engine=ocr_engine, use_gpu=use_gpu, custom_ocr_model_path=custom_ocr_model_path)
    if ocr_reader is None:
        logger.error("[ERROR] OCR reader not available")
        return []

    plates = []
    vehicle_crops = []
    vehicle_coords = []

    # Extract vehicle crops
    for idx, box in enumerate(car_boxes):
        x1, y1, x2, y2 = box
        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height or x1 >= x2 or y1 >= y2:
            logger.warning(f"[WARNING] Invalid vehicle box {idx+1}: ({x1}, {y1}, {x2}, {y2}), skipping")
            continue

        vehicle_crop = img[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            logger.warning(f"[WARNING] Empty vehicle crop at box {idx+1}, skipping")
            continue

        vehicle_crops.append(vehicle_crop)
        vehicle_coords.append((x1, y1, x2, y2))

    # Step 1: Detect plates
    plate_candidates = []
    for idx, (crop, (x1, y1, x2, y2)) in enumerate(zip(vehicle_crops, vehicle_coords)):
        plate_boxes = []
        plate_confidences = []
        if plate_detector:
            try:
                results = plate_detector.predict(crop, conf=conf_threshold, device='cuda' if use_gpu and torch.cuda.is_available() else 'cpu')
                for det in results:
                    boxes = det.boxes.xyxy.cpu().numpy()
                    confs = det.boxes.conf.cpu().numpy()
                    for box, conf in zip(boxes, confs):
                        px1, py1, px2, py2 = map(int, box)
                        # Map back to original image coordinates
                        px1, py1, px2, py2 = px1 + x1, py1 + y1, px2 + x1, py2 + y1
                        if (px1 < x1 or py1 < y1 or px2 > x2 or py2 > y2 or
                            px2 - px1 <= 10 or py2 - py1 <= 10):
                            continue
                        plate_boxes.append((px1, py1, px2, py2))
                        plate_confidences.append(float(conf))
                logger.info(f"[INFO] YOLOv8 detected {len(plate_boxes)} plates in vehicle box {idx+1}")
            except Exception as e:
                logger.error(f"[ERROR] YOLOv8 plate detection failed for vehicle box {idx+1}: {e}")

        # Step 2: Fallback to contour detection if YOLOv8 fails or is unavailable
        if not plate_boxes and plate_detector is not None:
            logger.info(f"[INFO] Falling back to contour detection for vehicle box {idx+1}")
            fallback_boxes = detect_plates_with_contours(img, (x1, y1, x2, y2))
            for box in fallback_boxes:
                plate_boxes.append(box)
                plate_confidences.append(0.5)  # Default confidence for contour detection

        # Step 3: If no plates detected, skip to next vehicle
        if not plate_boxes:
            logger.info(f"[INFO] No plates detected for vehicle box {idx+1}, skipping")
            continue

        for plate_box, plate_conf in zip(plate_boxes, plate_confidences):
            plate_candidates.append((plate_box, plate_conf, idx))

    # Step 4: Batch preprocess plate regions for OCR
    plate_crops = []
    plate_coords = []
    for (plate_x1, plate_y1, plate_x2, plate_y2), plate_conf, vehicle_idx in plate_candidates:
        x1, y1, _, _ = vehicle_coords[vehicle_idx]
        crop = vehicle_crops[vehicle_idx]
        rel_x1 = max(0, plate_x1 - x1)
        rel_y1 = max(0, plate_y1 - y1)
        rel_x2 = min(crop.shape[1], plate_x2 - x1)
        rel_y2 = min(crop.shape[0], plate_y2 - y1)
        if rel_x2 <= rel_x1 or rel_y2 <= rel_y1:
            logger.warning(f"[WARNING] Invalid plate crop region: ({rel_x1}, {rel_y1}, {rel_x2}, {rel_y2}), skipping")
            continue
        plate_crop = crop[rel_y1:rel_y2, rel_x1:rel_x2]
        processed_crop = preprocess_image_for_ocr(plate_crop)
        if processed_crop is None:
            continue
        plate_crops.append((processed_crop, plate_crop))  # Store both processed and original for retry
        plate_coords.append((plate_x1, plate_y1, plate_x2, plate_y2, plate_conf, vehicle_idx))

    # Step 5: Batch OCR processing with retry mechanism
    ocr_results = []
    if plate_crops:
        try:
            # First attempt
            processed_crops = [crop[0] for crop in plate_crops]
            if ocr_engine == "easyocr":
                ocr_results = ocr_reader.readtext_batched(processed_crops, detail=1) if hasattr(ocr_reader, 'readtext_batched') else [ocr_reader.readtext(crop, detail=1) for crop in processed_crops]
            else:  # Tesseract
                for crop in processed_crops:
                    result = ocr_reader.image_to_data(crop, output_type=ocr_reader.Output.DICT)
                    crop_results = []
                    for i in range(len(result['text'])):
                        text = result['text'][i].strip()
                        if text and int(result['conf'][i]) > 0:
                            conf = int(result['conf'][i]) / 100.0
                            x_ocr = result['left'][i]
                            y_ocr = result['top'][i]
                            w_ocr = result['width'][i]
                            h_ocr = result['height'][i]
                            bbox = [[x_ocr, y_ocr], [x_ocr + w_ocr, y_ocr], [x_ocr + w_ocr, y_ocr + h_ocr], [x_ocr, y_ocr + h_ocr]]
                            crop_results.append((bbox, text, conf))
                    ocr_results.append(crop_results)
            logger.info(f"[INFO] Performed batch OCR on {len(plate_crops)} plate regions")

            # Retry for failed OCR results
            retry_crops = []
            retry_indices = []
            for idx, (crop_pair, result) in enumerate(zip(plate_crops, ocr_results)):
                valid_result = False
                for _, text, prob in result:
                    if prob >= conf_threshold and validate_plate_text(text, region):
                        valid_result = True
                        break
                if not valid_result:
                    retry_crops.append(crop_pair[1])  # Use original crop for retry
                    retry_indices.append(idx)

            if retry_crops:
                logger.info(f"[INFO] Retrying OCR on {len(retry_crops)} regions with alternative preprocessing")
                retry_processed = [preprocess_image_for_ocr(crop, retry=True) for crop in retry_crops]
                retry_results = []
                if ocr_engine == "easyocr":
                    retry_results = ocr_reader.readtext_batched(retry_processed, detail=1) if hasattr(ocr_reader, 'readtext_batched') else [ocr_reader.readtext(crop, detail=1) for crop in retry_processed]
                else:  # Tesseract
                    for crop in retry_processed:
                        result = ocr_reader.image_to_data(crop, output_type=ocr_reader.Output.DICT)
                        crop_results = []
                        for i in range(len(result['text'])):
                            text = result['text'][i].strip()
                            if text and int(result['conf'][i]) > 0:
                                conf = int(result['conf'][i]) / 100.0
                                x_ocr = result['left'][i]
                                y_ocr = result['top'][i]
                                w_ocr = result['width'][i]
                                h_ocr = result['height'][i]
                                bbox = [[x_ocr, y_ocr], [x_ocr + w_ocr, y_ocr], [x_ocr + w_ocr, y_ocr + h_ocr], [x_ocr, y_ocr + h_ocr]]
                                crop_results.append((bbox, text, conf))
                        retry_results.append(crop_results)
                # Update ocr_results with retry results
                for idx, retry_result in zip(retry_indices, retry_results):
                    ocr_results[idx] = retry_result
                logger.info(f"[INFO] Completed OCR retry on {len(retry_crops)} regions")

        except Exception as e:
            logger.error(f"[ERROR] Batch OCR failed: {e}")

    # Step 6: Process OCR results
    for (plate_x1, plate_y1, plate_x2, plate_y2, plate_conf, vehicle_idx), crop_results in zip(plate_coords, ocr_results):
        plate_text = "UNKNOWN"
        ocr_conf = 0.0
        for (bbox, text, prob) in crop_results:
            if prob < conf_threshold:
                continue
            text = text.strip()
            if len(text) < 3 or not any(c.isalnum() for c in text):
                logger.debug(f"[DEBUG] Plate text too short or invalid: '{text}'")
                continue
            if validate_plate_text(text, region):
                plate_text = text
                ocr_conf = prob
                break
            else:
                logger.debug(f"[DEBUG] OCR result invalid for region: '{text}'")

        plates.append({
            "plate_text": plate_text,
            "bbox": (plate_x1, plate_y1, plate_x2, plate_y2),
            "plate_confidence": plate_conf,
            "ocr_confidence": ocr_conf,
            "vehicle_idx": vehicle_idx
        })
        logger.info(f"[INFO] Detected license plate: '{plate_text}' at ({plate_x1}, {plate_y1}, {plate_x2}, {plate_y2}) "
                    f"(plate conf: {plate_conf:.2f}, OCR conf: {ocr_conf:.2f})")

    # Save detected plates if requested
    if save_dir and plates:
        try:
            os.makedirs(save_dir, exist_ok=True)
            for idx, plate in enumerate(plates):
                x1, y1, x2, y2 = plate["bbox"]
                plate_img = img[y1:y2, x1:x2]
                if plate_img.size == 0:
                    logger.warning(f"[WARNING] Empty plate image at: ({x1}, {y1}, {x2}, {y2}), skipping save")
                    continue
                plate_text = plate["plate_text"].replace(" ", "_")
                plate_filename = os.path.join(save_dir, f"plate_{idx+1}_{plate_text}.jpg")
                cv2.imwrite(plate_filename, plate_img)
                logger.info(f"[INFO] Saved plate image to {plate_filename}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to save plate images: {e}")

    # Log summary
    valid_plates = sum(1 for plate in plates if plate["plate_text"] != "UNKNOWN")
    logger.info(f"[INFO] Total plates detected: {len(plates)}, Valid plates: {valid_plates}")
    if not plates:
        logger.warning("[WARNING] No plates detected")
    else:
        for idx, plate in enumerate(plates):
            logger.info(f"[INFO] Plate {idx+1}: '{plate['plate_text']}' at {plate['bbox']} "
                        f"(plate conf: {plate['plate_confidence']:.2f}, OCR conf: {plate['ocr_confidence']:.2f})")

    return plates


class PlateRecognizer:
    """
    Class for license plate recognition using OCR.

    This class provides methods for detecting and recognizing license plates in images
    using OCR techniques (EasyOCR or Tesseract).
    """

    def __init__(self, ocr_engine="easyocr", use_gpu=False, custom_ocr_model_path=None, region="India", existing_reader=None):
        """
        Initialize the PlateRecognizer.

        Args:
            ocr_engine: OCR engine to use ("easyocr" or "tesseract")
            use_gpu: Whether to use GPU for OCR
            custom_ocr_model_path: Path to a custom OCR model (applies to EasyOCR)
            region: Region for plate text validation (e.g., "India", "US", "EU")
            existing_reader: Preloaded OCR reader to avoid reloading (optional)
        """
        self.ocr_engine = ocr_engine
        self.use_gpu = use_gpu
        self.custom_ocr_model_path = custom_ocr_model_path
        self.region = region

        # Use existing reader if provided, otherwise load a new one
        if existing_reader is not None:
            self.reader = existing_reader
            logger.info("[INFO] Using provided OCR reader")
        else:
            self.reader = load_ocr_reader(ocr_engine=ocr_engine, use_gpu=use_gpu, custom_ocr_model_path=custom_ocr_model_path)
            if self.reader is None:
                logger.error(f"[ERROR] Failed to initialize {ocr_engine} reader")
                raise ValueError(f"Failed to initialize {ocr_engine} reader")

    def recognize(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Recognize license plate text in an image.

        Args:
            image: Input image (BGR format)

        Returns:
            Dictionary with plate text and OCR confidence, or None if no plate is found
        """
        if image is None or image.size == 0:
            logger.error("[ERROR] Invalid image provided for plate recognition")
            return None

        # Preprocess image for OCR
        processed_image = preprocess_image_for_ocr(image)
        if processed_image is None:
            return None

        try:
            if self.ocr_engine == "easyocr":
                results = self.reader.readtext(processed_image, detail=1)
                for (bbox, text, prob) in results:
                    if prob < 0.3:  # Minimum confidence threshold
                        continue
                    text = text.strip()
                    if len(text) < 3 or not any(c.isalnum() for c in text):
                        continue
                    if validate_plate_text(text, self.region):
                        return {"plate_text": text, "ocr_confidence": prob}
                # Retry with alternative preprocessing
                processed_image = preprocess_image_for_ocr(image, retry=True)
                if processed_image is None:
                    return None
                results = self.reader.readtext(processed_image, detail=1)
                for (bbox, text, prob) in results:
                    if prob < 0.3:
                        continue
                    text = text.strip()
                    if len(text) < 3 or not any(c.isalnum() for c in text):
                        continue
                    if validate_plate_text(text, self.region):
                        return {"plate_text": text, "ocr_confidence": prob}
            else:  # Tesseract
                result = self.reader.image_to_data(processed_image, output_type=self.reader.Output.DICT)
                for i in range(len(result['text'])):
                    text = result['text'][i].strip()
                    if text and int(result['conf'][i]) > 30:  # Minimum confidence threshold
                        conf = int(result['conf'][i]) / 100.0
                        if validate_plate_text(text, self.region):
                            return {"plate_text": text, "ocr_confidence": conf}
                # Retry with alternative preprocessing
                processed_image = preprocess_image_for_ocr(image, retry=True)
                if processed_image is None:
                    return None
                result = self.reader.image_to_data(processed_image, output_type=self.reader.Output.DICT)
                for i in range(len(result['text'])):
                    text = result['text'][i].strip()
                    if text and int(result['conf'][i]) > 30:
                        conf = int(result['conf'][i]) / 100.0
                        if validate_plate_text(text, self.region):
                            return {"plate_text": text, "ocr_confidence": conf}

            return None
        except Exception as e:
            logger.error(f"[ERROR] OCR recognition failed: {e}")
            return None

    def extract_plates(self, image: np.ndarray, vehicle_boxes: List[Tuple[int, int, int, int]], conf_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Extract license plates from the given image within vehicle bounding boxes.

        Args:
            image: Input image (BGR format)
            vehicle_boxes: List of vehicle bounding boxes (x1, y1, x2, y2)
            conf_threshold: Confidence threshold for plate detection and OCR

        Returns:
            List of dictionaries containing plate text, bounding box, plate confidence, and OCR confidence
        """
        return extract_license_plates(
            image_path=image,
            car_boxes=vehicle_boxes,
            ocr_engine=self.ocr_engine,
            conf_threshold=conf_threshold,
            use_gpu=self.use_gpu,
            custom_ocr_model_path=self.custom_ocr_model_path,
            region=self.region,
            ocr_reader=self.reader
        )