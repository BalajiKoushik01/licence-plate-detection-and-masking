"""
Image-based Vehicle Number Plate Agent
======================================

This module provides an agent for detecting and masking vehicle number plates
in static images using AI-based segmentation/detection. After masking, it
superimposes a branding logo onto the number plate with photorealistic realism,
handling plate orientation, lighting, shadows, and curvature.
"""

import os
import cv2
import logging
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from PIL import Image, ImageEnhance, ImageFilter

# Import PlateRecognizer
from vehicle_detection.plate_recognizer import PlateRecognizer
from vehicle_detection.detector import detect_vehicles
from vehicle_detection.plate_replacer import overlay_logo_on_plate

# Configure logging
logger = logging.getLogger(__name__)

def analyze_plate_perspective(image: np.ndarray, plate_box: Tuple[int, int, int, int]) -> Dict[str, Any]:
    """
    Analyze the perspective and lighting of a license plate for realistic logo overlay.

    Args:
        image: Input image as numpy array
        plate_box: Bounding box of the license plate (x1, y1, x2, y2)

    Returns:
        Dictionary containing perspective and lighting information
    """
    try:
        x_min, y_min, x_max, y_max = plate_box
        if x_min < 0 or y_min < 0 or x_max > image.shape[1] or y_max > image.shape[0] or x_min >= x_max or y_min >= y_max:
            logger.warning(f"[WARNING] Invalid plate box coordinates: {plate_box}, skipping analysis")
            return {
                'perspective': {'width': 0, 'height': 0, 'aspect_ratio': 1.0, 'is_skewed': False, 'rotation_angle': 0},
                'lighting': {'brightness': 128, 'contrast': 50, 'edge_density': 0, 'has_shadows': False}
            }

        # Calculate plate dimensions
        plate_width = x_max - x_min
        plate_height = y_max - y_min

        # Extract plate region
        plate_region = image[y_min:y_max, x_min:x_max]
        if plate_region.size == 0:
            logger.warning("[WARNING] Empty plate region, using default values")
            return {
                'perspective': {'width': plate_width, 'height': plate_height, 'aspect_ratio': 1.0, 'is_skewed': False, 'rotation_angle': 0},
                'lighting': {'brightness': 128, 'contrast': 50, 'edge_density': 0, 'has_shadows': False}
            }

        # Analyze perspective (simplified using edge detection for skew)
        gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray_plate, 100, 200)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)

        rotation_angle = 0
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
            if angles:
                rotation_angle = np.mean(angles)

        perspective = {
            'width': plate_width,
            'height': plate_height,
            'aspect_ratio': plate_width / max(1, plate_height),
            'is_skewed': abs(rotation_angle) > 5,  # Consider skewed if angle > 5 degrees
            'rotation_angle': rotation_angle
        }

        # Analyze lighting
        brightness = np.mean(gray_plate)
        contrast = np.std(gray_plate)
        edge_density = np.sum(edges > 0) / (plate_width * plate_height)

        lighting = {
            'brightness': brightness,
            'contrast': contrast,
            'edge_density': edge_density,
            'has_shadows': bool(edge_density > 0.1)
        }

        return {
            'perspective': perspective,
            'lighting': lighting
        }

    except Exception as e:
        logger.error(f"[ERROR] Failed to analyze plate perspective: {e}")
        return {
            'perspective': {'width': 0, 'height': 0, 'aspect_ratio': 1.0, 'is_skewed': False, 'rotation_angle': 0},
            'lighting': {'brightness': 128, 'contrast': 50, 'edge_density': 0, 'has_shadows': False}
        }

class ImagePlateAgent:
    def __init__(self, ocr_engine: str = "easyocr", conf_threshold: float = 0.5, existing_reader=None):
        """
        Initialize the Image Plate Agent for license plate detection and processing.

        Args:
            ocr_engine: OCR engine to use for plate recognition ('easyocr' or 'tesseract').
            conf_threshold: Confidence threshold for vehicle detection.
            existing_reader: Existing OCR reader instance to reuse.
        """
        self.conf_threshold = conf_threshold
        self.ocr_engine = ocr_engine
        try:
            self.plate_recognizer = PlateRecognizer(ocr_engine=ocr_engine, existing_reader=existing_reader)
            logger.info("[SUCCESS] PlateRecognizer initialized in ImagePlateAgent")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize PlateRecognizer: {e}")
            raise

    def preprocess_image(self, image: np.ndarray, max_size: int = 640) -> np.ndarray:
        """
        Preprocess image by resizing to reduce computation while preserving aspect ratio.

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

    def process_image(
        self,
        image_path: str,
        logo_path: Optional[str] = None,
        output_path: Optional[str] = None,
        use_gpu: bool = False
    ) -> Optional[str]:
        """
        Process an image to detect vehicles, extract license plates, and optionally replace them with a logo.

        Args:
            image_path: Path to the input image.
            logo_path: Path to the logo image for plate replacement (optional).
            output_path: Path to save the processed image (optional).
            use_gpu: Whether to use GPU for vehicle detection.

        Returns:
            Path to the processed image, or None if processing fails.
        """
        logger.info(f"[INFO] Processing image: {os.path.basename(image_path)}")

        # Step 1: Validate inputs
        if not os.path.exists(image_path):
            logger.error(f"[ERROR] Image not found: {image_path}")
            return None

        if logo_path and not os.path.exists(logo_path):
            logger.warning(f"[WARNING] Logo not found: {logo_path}, proceeding without logo")
            logo_path = None

        # Define a temporary output path for vehicle detection
        temp_output_dir = "outputs/temp"
        os.makedirs(temp_output_dir, exist_ok=True)
        temp_output_path = os.path.join(
            temp_output_dir, f"{os.path.splitext(os.path.basename(image_path))[0]}_detected.jpg"
        )

        # Step 2: Detect vehicles
        try:
            boxes, vehicle_counts = detect_vehicles(
                image_path=image_path,
                save_path=temp_output_path,
                conf_threshold=self.conf_threshold,
                show=False,
                open_image=False,
                use_gpu=use_gpu
            )
            logger.info(f"[SUCCESS] Detected {vehicle_counts.get('Total', 0)} vehicles")
            for vehicle_type, count in vehicle_counts.items():
                if vehicle_type != "Total" and count > 0:
                    logger.info(f"  - {vehicle_type}: {count}")
        except Exception as e:
            logger.error(f"[ERROR] Vehicle detection failed: {e}")
            return None

        # Step 3: Extract license plates
        if not boxes:
            logger.warning("[WARNING] No vehicles detected for plate extraction")
            return temp_output_path

        # Load the detected image
        image = cv2.imread(temp_output_path)
        if image is None:
            logger.error(f"[ERROR] Failed to load detected image: {temp_output_path}")
            return None

        # Preprocess image for plate detection
        image = self.preprocess_image(image)

        # Convert boxes to the format expected by PlateRecognizer
        clean_boxes = [box['bbox'] for box in boxes]

        # Extract plates using PlateRecognizer
        try:
            plate_boxes = self.plate_recognizer.extract_plates(
                image=image,
                vehicle_boxes=clean_boxes,
                conf_threshold=0.01
            )
            logger.info(f"[SUCCESS] Detected {len(plate_boxes)} license plates")
        except Exception as e:
            logger.error(f"[ERROR] Plate extraction failed: {e}")
            return temp_output_path

        # Step 4: Replace plates with logo if requested
        if plate_boxes and logo_path:
            final_output_path = output_path or os.path.join(
                "outputs", f"{os.path.splitext(os.path.basename(image_path))[0]}_agent_processed.jpg"
            )
            os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

            # Extract coordinates for overlay
            plate_coords = [coords for _, coords in plate_boxes]

            try:
                # Overlay logo on each plate with photorealistic adjustments
                result_image = image
                for idx, plate_box in enumerate(plate_coords):
                    result_image = self.overlay_logo_photorealistic(result_image, plate_box, logo_path)
                    logger.info(f"[SUCCESS] Overlaid logo on plate {idx + 1}")

                # Save the final image
                cv2.imwrite(final_output_path, result_image)
                logger.info(f"[SUCCESS] Plates replaced. Output saved to: {final_output_path}")
                return final_output_path
            except Exception as e:
                logger.error(f"[ERROR] Logo overlay failed: {e}")
                return temp_output_path

        return temp_output_path

    def detect_plates(self, image_path: str, use_gpu: bool = False) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], List[List[Tuple[int, int]]]]:
        """
        Detect license plates in an image.

        Args:
            image_path: Path to the input image.
            use_gpu: Whether to use GPU for vehicle detection.

        Returns:
            Tuple containing:
            - The processed image as a NumPy array.
            - List of plate bounding boxes (x1, y1, x2, y2).
            - List of vehicle bounding boxes as lists of (x, y) coordinates.
        """
        logger.info(f"[INFO] Detecting plates in image: {os.path.basename(image_path)}")

        # Step 1: Validate input
        if not os.path.exists(image_path):
            logger.error(f"[ERROR] Image not found: {image_path}")
            return None, [], []

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"[ERROR] Failed to load image: {image_path}")
            return None, [], []

        # Step 2: Detect vehicles
        try:
            boxes, vehicle_counts = detect_vehicles(
                image_path=image_path,
                save_path=None,
                conf_threshold=self.conf_threshold,
                show=False,
                open_image=False,
                use_gpu=use_gpu
            )
            logger.info(f"[SUCCESS] Detected {vehicle_counts.get('Total', 0)} vehicles")
        except Exception as e:
            logger.error(f"[ERROR] Vehicle detection failed: {e}")
            return image, [], []

        if not boxes:
            logger.warning("[WARNING] No vehicles detected for plate extraction")
            return image, [], []

        # Step 3: Extract license plates
        clean_boxes = [box['bbox'] for box in boxes]
        vehicle_coords = [
            [(box[0], box[1]), (box[2], box[1]), (box[2], box[3]), (box[0], box[3])]
            for box in clean_boxes
        ]

        try:
            plate_boxes = self.plate_recognizer.extract_plates(
                image=image,
                vehicle_boxes=clean_boxes,
                conf_threshold=0.01
            )
            logger.info(f"[SUCCESS] Detected {len(plate_boxes)} license plates")
        except Exception as e:
            logger.error(f"[ERROR] Plate extraction failed: {e}")
            return image, [], vehicle_coords

        plate_coords = [coords for _, coords in plate_boxes]
        return image, plate_coords, vehicle_coords

    def overlay_logo_photorealistic(self, image: np.ndarray, plate_box: Tuple[int, int, int, int], logo_path: str) -> np.ndarray:
        """
        Overlay a logo on a license plate with photorealistic realism, adjusting for perspective and lighting.

        Args:
            image: Input image as numpy array
            plate_box: Bounding box of the license plate (x1, y1, x2, y2)
            logo_path: Path to the logo image

        Returns:
            Image with the logo overlaid on the license plate
        """
        # Step 1: Load logo
        logo_path = os.path.normpath(logo_path)
        if not os.path.exists(logo_path):
            logger.error(f"[ERROR] Logo file not found: {logo_path}")
            return image

        try:
            logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo_img is None:
                logger.error(f"[ERROR] Failed to load logo: {logo_path}")
                return image
        except Exception as e:
            logger.error(f"[ERROR] Error loading logo: {e}")
            return image

        # Step 2: Analyze plate perspective and lighting
        plate_analysis = analyze_plate_perspective(image, plate_box)
        if plate_analysis['perspective']['width'] == 0 or plate_analysis['perspective']['height'] == 0:
            logger.warning("[WARNING] Invalid plate analysis, skipping logo overlay")
            return image

        # Step 3: Extract plate region
        x_min, y_min, x_max, y_max = plate_box
        plate_width = x_max - x_min
        plate_height = y_max - y_min
        plate_region = image[y_min:y_max, x_min:x_max].copy()

        # Step 4: Resize logo to fit plate region
        try:
            resized_logo = cv2.resize(logo_img, (plate_width, plate_height), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.error(f"[ERROR] Failed to resize logo: {e}")
            return image

        # Step 5: Apply perspective transformation if needed
        if plate_analysis['perspective']['is_skewed']:
            angle = plate_analysis['perspective']['rotation_angle']
            center = (plate_width // 2, plate_height // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            try:
                resized_logo = cv2.warpAffine(resized_logo, matrix, (plate_width, plate_height))
            except Exception as e:
                logger.warning(f"[WARNING] Failed to apply perspective transformation: {e}")

        # Step 6: Adjust logo brightness and contrast to match plate
        try:
            if len(resized_logo.shape) > 2 and resized_logo.shape[2] >= 3:
                logo_pil = Image.fromarray(
                    cv2.cvtColor(resized_logo, cv2.COLOR_BGRA2RGBA) if resized_logo.shape[2] == 4 else cv2.cvtColor(resized_logo, cv2.COLOR_BGR2RGB)
                )

                # Adjust brightness
                brightness_factor = plate_analysis['lighting']['brightness'] / 128
                logo_pil = ImageEnhance.Brightness(logo_pil).enhance(brightness_factor)

                # Adjust contrast
                contrast_factor = plate_analysis['lighting']['contrast'] / 50
                logo_pil = ImageEnhance.Contrast(logo_pil).enhance(contrast_factor)

                # Add slight blur to match plate quality
                logo_pil = logo_pil.filter(ImageFilter.GaussianBlur(radius=0.5))

                # Convert back to numpy array
                if resized_logo.shape[2] == 4:
                    resized_logo = cv2.cvtColor(np.array(logo_pil), cv2.COLOR_RGBA2BGRA)
                else:
                    resized_logo = cv2.cvtColor(np.array(logo_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"[WARNING] Failed to adjust logo lighting: {e}")

        # Step 7: Apply the enhanced logo to the plate region
        result_image = image.copy()
        has_alpha = resized_logo.shape[2] == 4 if len(resized_logo.shape) > 2 else False

        try:
            if has_alpha:
                roi = result_image[y_min:y_max, x_min:x_max]
                alpha_channel = resized_logo[:, :, 3] / 255.0
                alpha_3channel = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=2)
                logo_bgr = resized_logo[:, :, 0:3]
                blended = (logo_bgr * alpha_3channel) + (roi * (1 - alpha_3channel))
                result_image[y_min:y_max, x_min:x_max] = blended.astype(np.uint8)
            else:
                result_image[y_min:y_max, x_min:x_max] = resized_logo
        except Exception as e:
            logger.error(f"[ERROR] Failed to overlay logo on plate: {e}")
            return image

        return result_image

    def mask_plate_with_logo(self, image: np.ndarray, plate_box: Tuple[int, int, int, int], logo_path: str) -> np.ndarray:
        """
        Mask a license plate with a logo as a fallback method.

        Args:
            image: Input image as numpy array
            plate_box: Bounding box of the license plate (x1, y1, x2, y2)
            logo_path: Path to the logo image

        Returns:
            Image with the plate masked
        """
        if not logo_path or not os.path.exists(logo_path):
            logger.warning("[WARNING] Logo path invalid, applying blur mask instead")
            x_min, y_min, x_max, y_max = plate_box
            roi = image[y_min:y_max, x_min:x_max]
            blurred = cv2.GaussianBlur(roi, (15, 15), 0)
            image[y_min:y_max, x_min:x_max] = blurred
            return image

        try:
            logo_img = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo_img is None:
                logger.error(f"[ERROR] Failed to load logo: {logo_path}")
                return image
        except Exception as e:
            logger.error(f"[ERROR] Error loading logo: {e}")
            return image

        x_min, y_min, x_max, y_max = plate_box
        plate_width = x_max - x_min
        plate_height = y_max - y_min

        try:
            resized_logo = cv2.resize(logo_img, (plate_width, plate_height), interpolation=cv2.INTER_AREA)
            result_image = image.copy()
            result_image[y_min:y_max, x_min:x_max] = resized_logo
            return result_image
        except Exception as e:
            logger.error(f"[ERROR] Failed to mask plate with logo: {e}")
            return image