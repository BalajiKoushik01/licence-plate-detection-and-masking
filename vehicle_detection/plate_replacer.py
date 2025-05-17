"""
License Plate Replacer Module
============================

This module provides functionality for overlaying logos on license plates in images and videos,
with photorealistic adjustments for perspective, lighting, and shadows.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Union
from PIL import Image, ImageEnhance, ImageFilter
import os

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

class PlateReplacer:
    """Class for overlaying logos on license plates with photorealistic effects."""

    def __init__(self, use_gpu: bool = False, max_size: int = 640):
        """
        Initialize the PlateReplacer.

        Args:
            use_gpu: Whether to use GPU for image processing (if available).
            max_size: Maximum size for image dimensions to optimize performance.
        """
        self.use_gpu = use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0
        self.max_size = max_size
        if self.use_gpu:
            logger.info("[INFO] GPU acceleration enabled for image processing")
        else:
            logger.info("[INFO] Using CPU for image processing")

    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess the image by resizing to reduce computation while preserving aspect ratio.

        Args:
            image: Input image as numpy array

        Returns:
            Preprocessed image or None if preprocessing fails
        """
        try:
            h, w = image.shape[:2]
            scale = min(self.max_size / h, self.max_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            if self.use_gpu:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(image)
                gpu_resized = cv2.cuda.resize(gpu_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                return gpu_resized.download()
            return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.error(f"[ERROR] Failed to preprocess image: {e}")
            return None

    def analyze_plate_region(self, image: np.ndarray, plate_box: Tuple[int, int, int, int]) -> dict:
        """
        Analyze the plate region for lighting and perspective to adjust the logo.

        Args:
            image: Input image as numpy array
            plate_box: Bounding box of the license plate (x1, y1, x2, y2)

        Returns:
            Dictionary with lighting and perspective information
        """
        try:
            x1, y1, x2, y2 = plate_box
            if x1 < 0 or y1 < 0 or x2 > image.shape[1] or y2 > image.shape[0] or x1 >= x2 or y1 >= y2:
                logger.warning(f"[WARNING] Invalid plate box coordinates: {plate_box}")
                return {
                    'brightness': 128,
                    'contrast': 50,
                    'has_shadows': False,
                    'shadow_direction': (0, 0),
                    'transform_matrix': None,
                    'rotation_angle': 0
                }

            plate_region = image[y1:y2, x1:x2]
            if plate_region.size == 0:
                logger.warning("[WARNING] Empty plate region")
                return {
                    'brightness': 128,
                    'contrast': 50,
                    'has_shadows': False,
                    'shadow_direction': (0, 0),
                    'transform_matrix': None,
                    'rotation_angle': 0
                }

            # Analyze lighting
            gray_plate = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray_plate)
            contrast = np.std(gray_plate)
            edges = cv2.Canny(gray_plate, 100, 200)
            edge_density = np.sum(edges > 0) / (plate_region.shape[0] * plate_region.shape[1])

            # Estimate shadow direction
            shadow_direction = (0, 0)
            if edge_density > 0.1:
                grad_x = cv2.Sobel(gray_plate, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_plate, cv2.CV_64F, 0, 1, ksize=3)
                shadow_direction = (np.mean(grad_x), np.mean(grad_y))

            # Analyze perspective
            plate_width = x2 - x1
            plate_height = y2 - y1
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=20, maxLineGap=10)
            rotation_angle = 0
            transform_matrix = None
            if lines is not None:
                angles = []
                for line in lines:
                    x1_line, y1_line, x2_line, y2_line = line[0]
                    angle = np.arctan2(y2_line - y1_line, x2_line - x1_line) * 180 / np.pi
                    angles.append(angle)
                if angles:
                    rotation_angle = np.mean(angles)
                    src_points = np.float32([[0, 0], [plate_width, 0], [plate_width, plate_height], [0, plate_height]])
                    dst_points = np.float32([
                        [0, 0],
                        [plate_width, 0],
                        [plate_width - 5 * np.sin(np.radians(rotation_angle)), plate_height],
                        [5 * np.sin(np.radians(rotation_angle)), plate_height]
                    ])
                    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

            return {
                'brightness': brightness,
                'contrast': contrast,
                'has_shadows': edge_density > 0.1,
                'shadow_direction': shadow_direction,
                'transform_matrix': transform_matrix,
                'rotation_angle': rotation_angle
            }
        except Exception as e:
            logger.error(f"[ERROR] Failed to analyze plate region: {e}")
            return {
                'brightness': 128,
                'contrast': 50,
                'has_shadows': False,
                'shadow_direction': (0, 0),
                'transform_matrix': None,
                'rotation_angle': 0
            }

    def create_shadow(self, logo: np.ndarray, width: int, height: int, shadow_direction: Tuple[float, float]) -> np.ndarray:
        """
        Create a shadow effect for the logo.

        Args:
            logo: Logo image as numpy array
            width: Width of the target region
            height: Height of the target region
            shadow_direction: Direction of the shadow (dx, dy)

        Returns:
            Shadow image as numpy array
        """
        try:
            dx, dy = shadow_direction
            shadow_offset = (int(dx * 0.1), int(dy * 0.1))
            shadow = np.zeros((height, width, 4), dtype=np.uint8)

            if logo.shape[2] == 4:
                alpha = logo[:, :, 3]
                shadow_base = cv2.GaussianBlur(alpha, (5, 5), 0)
                shadow_base = (shadow_base * 0.5).astype(np.uint8)
                shadow[:, :, 3] = shadow_base
                shadow[:, :, 0:3] = (50, 50, 50)  # Dark gray shadow
                M = np.float32([[1, 0, shadow_offset[0]], [0, 1, shadow_offset[1]]])
                shadow = cv2.warpAffine(shadow, M, (width, height))
            return shadow
        except Exception as e:
            logger.warning(f"[WARNING] Failed to create shadow: {e}")
            return np.zeros((height, width, 4), dtype=np.uint8)

    def overlay_logo_on_plate(
        self,
        image_path: str,
        plate_boxes: List[Tuple[int, int, int, int]],
        logo_path: str,
        save_path: str
    ) -> Optional[np.ndarray]:
        """
        Overlay a logo on detected license plates in an image with photorealistic effects.

        Args:
            image_path: Path to the input image.
            plate_boxes: List of plate bounding boxes (x1, y1, x2, y2).
            logo_path: Path to the logo image (with transparency).
            save_path: Path to save the processed image.

        Returns:
            Processed image as a NumPy array, or None if processing fails.
        """
        # Load and validate logo
        logo_path = os.path.normpath(logo_path)
        if not os.path.exists(logo_path):
            logger.error(f"[ERROR] Logo file not found: {logo_path}")
            return None
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            logger.error(f"[ERROR] Failed to load logo from: {logo_path}")
            return None
        if logo.shape[2] == 3:
            logger.debug("[DEBUG] Logo has no alpha channel, adding one")
            alpha_channel = np.ones((logo.shape[0], logo.shape[1], 1), dtype=logo.dtype) * 255
            logo = np.concatenate((logo, alpha_channel), axis=2)
        elif logo.shape[2] != 4:
            logger.error(f"[ERROR] Logo must have 4 channels (RGBA), but has {logo.shape[2]}")
            return None
        logo_height, logo_width = logo.shape[:2]
        if logo_width <= 0 or logo_height <= 0:
            logger.error(f"[ERROR] Invalid logo dimensions: {logo_width}x{logo_height}")
            return None

        # Load and validate image
        if not os.path.exists(image_path):
            logger.error(f"[ERROR] Image file not found: {image_path}")
            return None
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"[ERROR] Failed to load image from: {image_path}")
            return None
        if image.shape[2] != 3:
            logger.error(f"[ERROR] Image must have 3 channels (RGB), but has {image.shape[2]}")
            return None
        img_height, img_width = image.shape[:2]
        if img_width <= 0 or img_height <= 0:
            logger.error(f"[ERROR] Invalid image dimensions: {img_width}x{img_height}")
            return None

        # Preprocess image
        image = self.preprocess_image(image)
        if image is None:
            return None

        # Validate plate boxes
        if not isinstance(plate_boxes, list) or not all(isinstance(box, tuple) and len(box) == 4 for box in plate_boxes):
            logger.error("[ERROR] Invalid plate boxes format, expected list of tuples (x1, y1, x2, y2)")
            return None
        if len(plate_boxes) == 0:
            logger.warning("[WARNING] No plate boxes provided for logo overlay")
            cv2.imwrite(save_path, image)
            logger.info(f"[SUCCESS] Output image saved to: {save_path}")
            return image
        if len(plate_boxes) > 50:
            logger.error(f"[ERROR] Excessive plate boxes found: {len(plate_boxes)}, processing will be skipped")
            return None

        result_image = image.copy()
        for idx, (x1, y1, x2, y2) in enumerate(plate_boxes, 1):
            logger.info(f"[INFO] Processing plate box {idx}/{len(plate_boxes)}: ({x1}, {y1}, {x2}, {y2})")
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height or x1 >= x2 or y1 >= y2:
                logger.warning(f"[WARNING] Invalid plate coordinates: ({x1}, {y1}, {x2}, {y2}), skipping")
                continue

            # Analyze plate region
            analysis = self.analyze_plate_region(image, (x1, y1, x2, y2))
            plate_width = x2 - x1
            plate_height = y2 - y1

            # Resize logo
            try:
                resized_logo = cv2.resize(logo, (plate_width, plate_height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                logger.error(f"[ERROR] Failed to resize logo: {e}")
                continue

            # Apply perspective transformation
            if analysis['transform_matrix'] is not None:
                try:
                    resized_logo = cv2.warpPerspective(resized_logo, analysis['transform_matrix'], (plate_width, plate_height))
                except Exception as e:
                    logger.warning(f"[WARNING] Failed to apply perspective transformation: {e}")

            # Adjust logo lighting
            try:
                logo_pil = Image.fromarray(
                    cv2.cvtColor(resized_logo, cv2.COLOR_BGRA2RGBA) if resized_logo.shape[2] == 4 else cv2.cvtColor(resized_logo, cv2.COLOR_BGR2RGB)
                )
                brightness_factor = analysis['brightness'] / 128
                contrast_factor = analysis['contrast'] / 50
                logo_pil = ImageEnhance.Brightness(logo_pil).enhance(brightness_factor)
                logo_pil = ImageEnhance.Contrast(logo_pil).enhance(contrast_factor)
                logo_pil = logo_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
                resized_logo = cv2.cvtColor(np.array(logo_pil), cv2.COLOR_RGBA2BGRA)
            except Exception as e:
                logger.warning(f"[WARNING] Failed to adjust logo lighting: {e}")

            # Create shadow
            shadow = None
            if analysis['has_shadows']:
                shadow = self.create_shadow(resized_logo, plate_width, plate_height, analysis['shadow_direction'])

            # Overlay logo and shadow
            try:
                roi = result_image[y1:y2, x1:x2]
                if analysis['has_shadows'] and shadow is not None:
                    shadow_alpha = shadow[:, :, 3] / 255.0
                    shadow_alpha_3channel = np.stack([shadow_alpha, shadow_alpha, shadow_alpha], axis=2)
                    shadow_bgr = shadow[:, :, 0:3]
                    roi = (shadow_bgr * shadow_alpha_3channel + roi * (1 - shadow_alpha_3channel)).astype(np.uint8)

                alpha_channel = resized_logo[:, :, 3] / 255.0
                alpha_3channel = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=2)
                logo_bgr = resized_logo[:, :, 0:3]
                blended = (logo_bgr * alpha_3channel + roi * (1 - alpha_3channel)).astype(np.uint8)
                result_image[y1:y2, x1:x2] = blended
            except Exception as e:
                logger.error(f"[ERROR] Failed to overlay logo: {e}")
                result_image[y1:y2, x1:x2] = self.adaptive_blur_plate(result_image[y1:y2, x1:x2])
                logger.info("[INFO] Applied adaptive blur as fallback")

        # Save the processed image
        try:
            cv2.imwrite(save_path, result_image)
            logger.info(f"[SUCCESS] Output image saved to: {save_path}")
            return result_image
        except Exception as e:
            logger.error(f"[ERROR] Failed to save output image: {e}")
            return None

    def overlay_logo_on_video(
        self,
        video_path: str,
        plate_boxes: List[List[Tuple[int, int]]],
        logo_path: str,
        output_path: str
    ) -> bool:
        """
        Overlay a logo on detected license plates in a video with temporal consistency.

        Args:
            video_path: Path to the input video.
            plate_boxes: List of plate bounding boxes as lists of (x, y) coordinates per frame.
            logo_path: Path to the logo image (with transparency).
            output_path: Path to save the processed video.

        Returns:
            True if processing succeeds, False otherwise.
        """
        # Load and validate logo
        logo_path = os.path.normpath(logo_path)
        if not os.path.exists(logo_path):
            logger.error(f"[ERROR] Logo file not found: {logo_path}")
            return False
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        if logo is None:
            logger.error(f"[ERROR] Failed to load logo from: {logo_path}")
            return False
        if logo.shape[2] == 3:
            logger.debug("[DEBUG] Logo has no alpha channel, adding one")
            alpha_channel = np.ones((logo.shape[0], logo.shape[1], 1), dtype=logo.dtype) * 255
            logo = np.concatenate((logo, alpha_channel), axis=2)
        elif logo.shape[2] != 4:
            logger.error(f"[ERROR] Logo must have 4 channels (RGBA), but has {logo.shape[2]}")
            return False

        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[ERROR] Failed to open video file: {video_path}")
            return False

        # Video writer setup with MP4 support
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if '.mp4' in output_path.lower():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        else:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logger.error(f"[ERROR] Failed to initialize video writer: {output_path}")
            cap.release()
            return False

        # Validate plate boxes
        if not isinstance(plate_boxes, list) or not all(isinstance(frame_boxes, list) for frame_boxes in plate_boxes):
            logger.error("[ERROR] Invalid plate boxes format, expected list of lists")
            cap.release()
            out.release()
            return False
        if len(plate_boxes) == 0:
            logger.warning("[WARNING] No plate boxes provided for logo overlay")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)
            cap.release()
            out.release()
            logger.info(f"[SUCCESS] Output video saved to: {output_path}")
            return True

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if len(plate_boxes) != frame_count:
            logger.warning(f"[WARNING] Plate boxes ({len(plate_boxes)}) do not match frame count ({frame_count}), adjusting")
            if len(plate_boxes) < frame_count:
                plate_boxes.extend([plate_boxes[-1]] * (frame_count - len(plate_boxes)))
            else:
                plate_boxes = plate_boxes[:frame_count]

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("[INFO] End of video reached")
                break

            frame = self.preprocess_image(frame)
            if frame is None:
                continue

            boxes = plate_boxes[frame_idx]
            for idx, box in enumerate(boxes, 1):
                if len(box) != 4:
                    logger.warning(f"[WARNING] Invalid plate box format at frame {frame_idx}, index {idx}: {box}, skipping")
                    continue
                # Convert box format to (x1, y1, x2, y2)
                x1 = min(p[0] for p in box)
                y1 = min(p[1] for p in box)
                x2 = max(p[0] for p in box)
                y2 = max(p[1] for p in box)
                frame = self.overlay_logo_on_frame(frame, [(x1, y1, x2, y2)], logo)

            out.write(frame)
            frame_idx += 1

        cap.release()
        out.release()
        logger.info(f"[SUCCESS] Output video saved to: {output_path}")
        return True

    def overlay_logo_on_frame(
        self,
        frame: np.ndarray,
        plate_boxes: List[Tuple[int, int, int, int]],
        logo: np.ndarray
    ) -> np.ndarray:
        """
        Overlay a logo on detected license plates in a video frame.

        Args:
            frame: Input frame as a NumPy array.
            plate_boxes: List of plate bounding boxes (x1, y1, x2, y2).
            logo: Logo image as a NumPy array (with transparency).

        Returns:
            Processed frame as a NumPy array.
        """
        if len(plate_boxes) == 0:
            logger.warning("[WARNING] No plate boxes provided for logo overlay on frame")
            return frame

        img_height, img_width = frame.shape[:2]
        for idx, (x1, y1, x2, y2) in enumerate(plate_boxes, 1):
            if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height or x1 >= x2 or y1 >= y2:
                logger.warning(f"[WARNING] Invalid plate coordinates: ({x1}, {y1}, {x2}, {y2}), skipping")
                continue

            # Analyze plate region
            analysis = self.analyze_plate_region(frame, (x1, y1, x2, y2))
            plate_width = x2 - x1
            plate_height = y2 - y1

            # Resize logo
            try:
                resized_logo = cv2.resize(logo, (plate_width, plate_height), interpolation=cv2.INTER_AREA)
            except Exception as e:
                logger.error(f"[ERROR] Failed to resize logo: {e}")
                frame[y1:y2, x1:x2] = self.adaptive_blur_plate(frame[y1:y2, x1:x2])
                continue

            # Apply perspective transformation
            if analysis['transform_matrix'] is not None:
                try:
                    resized_logo = cv2.warpPerspective(resized_logo, analysis['transform_matrix'], (plate_width, plate_height))
                except Exception as e:
                    logger.warning(f"[WARNING] Failed to apply perspective transformation: {e}")

            # Adjust logo lighting
            try:
                logo_pil = Image.fromarray(
                    cv2.cvtColor(resized_logo, cv2.COLOR_BGRA2RGBA) if resized_logo.shape[2] == 4 else cv2.cvtColor(resized_logo, cv2.COLOR_BGR2RGB)
                )
                brightness_factor = analysis['brightness'] / 128
                contrast_factor = analysis['contrast'] / 50
                logo_pil = ImageEnhance.Brightness(logo_pil).enhance(brightness_factor)
                logo_pil = ImageEnhance.Contrast(logo_pil).enhance(contrast_factor)
                logo_pil = logo_pil.filter(ImageFilter.GaussianBlur(radius=0.5))
                resized_logo = cv2.cvtColor(np.array(logo_pil), cv2.COLOR_RGBA2BGRA)
            except Exception as e:
                logger.warning(f"[WARNING] Failed to adjust logo lighting: {e}")

            # Create shadow
            shadow = None
            if analysis['has_shadows']:
                shadow = self.create_shadow(resized_logo, plate_width, plate_height, analysis['shadow_direction'])

            # Overlay logo and shadow
            try:
                roi = frame[y1:y2, x1:x2]
                if analysis['has_shadows'] and shadow is not None:
                    shadow_alpha = shadow[:, :, 3] / 255.0
                    shadow_alpha_3channel = np.stack([shadow_alpha, shadow_alpha, shadow_alpha], axis=2)
                    shadow_bgr = shadow[:, :, 0:3]
                    roi = (shadow_bgr * shadow_alpha_3channel + roi * (1 - shadow_alpha_3channel)).astype(np.uint8)

                alpha_channel = resized_logo[:, :, 3] / 255.0
                alpha_3channel = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=2)
                logo_bgr = resized_logo[:, :, 0:3]
                blended = (logo_bgr * alpha_3channel + roi * (1 - alpha_3channel)).astype(np.uint8)
                frame[y1:y2, x1:x2] = blended
            except Exception as e:
                logger.error(f"[ERROR] Failed to overlay logo on frame: {e}")
                frame[y1:y2, x1:x2] = self.adaptive_blur_plate(frame[y1:y2, x1:x2])

        return frame

    def mask_plate_with_logo(
        self,
        frame: np.ndarray,
        plate_boxes: List[Tuple[int, int, int, int]],
        logo: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Mask a detected license plate with a logo or a blur if no logo is provided.

        Args:
            frame: Input frame as a NumPy array.
            plate_boxes: List of plate bounding boxes (x1, y1, x2, y2).
            logo: Logo image as a NumPy array (with transparency), or None to blur.

        Returns:
            Processed frame as a NumPy array.
        """
        if len(plate_boxes) == 0:
            logger.warning("[WARNING] No plate boxes provided for masking")
            return frame

        for idx, (x1, y1, x2, y2) in enumerate(plate_boxes, 1):
            if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or x1 >= x2 or y1 >= y2:
                logger.warning(f"[WARNING] Invalid plate coordinates: ({x1}, {y1}, {x2}, {y2}), skipping")
                continue

            if logo is not None:
                frame = self.overlay_logo_on_frame(frame, [(x1, y1, x2, y2)], logo)
            else:
                frame[y1:y2, x1:x2] = self.adaptive_blur_plate(frame[y1:y2, x1:x2])
                logger.debug(f"[DEBUG] Applied adaptive blur at: ({x1}, {y1}, {x2}, {y2})")

        return frame

    def adaptive_blur_plate(self, region: np.ndarray) -> np.ndarray:
        """
        Apply an adaptive blur to a license plate region based on its size.

        Args:
            region: Plate region as a NumPy array.

        Returns:
            Blurred region as a NumPy array.
        """
        h, w = region.shape[:2]
        kernel_size = max(3, int(min(w, h) / 10) * 2 + 1)
        logger.debug(f"[DEBUG] Using blur kernel size: {kernel_size} for region {w}x{h}")
        return cv2.GaussianBlur(region, (kernel_size, kernel_size), 0)

def overlay_logo_on_plate(
    image_path: str,
    plate_boxes: List[Tuple[int, int, int, int]],
    logo_path: str,
    save_path: str,
    use_gpu: bool = False
) -> Optional[np.ndarray]:
    """
    Standalone function to overlay a logo on detected license plates in an image.

    Args:
        image_path: Path to the input image.
        plate_boxes: List of plate bounding boxes (x1, y1, x2, y2).
        logo_path: Path to the logo image (with transparency).
        save_path: Path to save the processed image.
        use_gpu: Whether to use GPU for processing.

    Returns:
        Processed image as a NumPy array, or None if processing fails.
    """
    replacer = PlateReplacer(use_gpu=use_gpu)
    return replacer.overlay_logo_on_plate(image_path, plate_boxes, logo_path, save_path)

def overlay_logo_on_video(
    video_path: str,
    plate_boxes: List[List[Tuple[int, int]]],
    logo_path: str,
    output_path: str,
    use_gpu: bool = False
) -> bool:
    """
    Standalone function to overlay a logo on detected license plates in a video.

    Args:
        video_path: Path to the input video.
        plate_boxes: List of plate bounding boxes as lists of (x, y) coordinates per frame.
        logo_path: Path to the logo image (with transparency).
        output_path: Path to save the processed video.
        use_gpu: Whether to use GPU for processing.

    Returns:
        True if processing succeeds, False otherwise.
    """
    replacer = PlateReplacer(use_gpu=use_gpu)
    return replacer.overlay_logo_on_video(video_path, plate_boxes, logo_path, output_path)

def overlay_logo_on_frame(
    frame: np.ndarray,
    plate_boxes: List[Tuple[int, int, int, int]],
    logo_img: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Standalone function to overlay a logo on detected license plates in a video frame.

    Args:
        frame: Input frame (BGR format)
        plate_boxes: List of bounding boxes for plates (x1, y1, x2, y2)
        logo_img: Logo image (optional, if None, uses adaptive blur)

    Returns:
        Processed frame with logos or blur overlaid
    """
    if frame is None or frame.size == 0:
        logger.error("[ERROR] Invalid frame provided")
        return frame

    # Validate logo
    if logo_img is not None:
        if logo_img.shape[2] == 3:
            logger.debug("[DEBUG] Logo has no alpha channel, adding one")
            alpha_channel = np.ones((logo_img.shape[0], logo_img.shape[1], 1), dtype=logo_img.dtype) * 255
            logo_img = np.concatenate((logo_img, alpha_channel), axis=2)
        elif logo_img.shape[2] != 4:
            logger.warning(f"[WARNING] Logo must have 4 channels (RGBA), but has {logo_img.shape[2]}. Using blur instead.")
            logo_img = None

    # Use PlateReplacer to process the frame
    replacer = PlateReplacer()
    return replacer.mask_plate_with_logo(frame, plate_boxes, logo_img)

if __name__ == "__main__":
    # Example usage
    image_path = "path/to/image.jpg"
    plate_boxes = [(100, 200, 300, 400), (150, 250, 350, 450)]
    logo_path = "path/to/logo.png"
    save_path = "path/to/output_image.jpg"

    overlay_logo_on_plate(image_path, plate_boxes, logo_path, save_path)

    video_path = "path/to/video.mp4"
    output_video_path = "path/to/output_video.mp4"
    plate_boxes_video = [[(100, 200), (300, 200), (300, 400), (100, 400)],
                        [(150, 250), (350, 250), (350, 450), (150, 450)]]

    overlay_logo_on_video(video_path, plate_boxes_video, logo_path, output_video_path)