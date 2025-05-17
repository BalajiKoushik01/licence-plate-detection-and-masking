"""
License Plate Utilities
======================

This module provides utility functions for license plate detection and masking.
"""

import cv2
import numpy as np
import os
import logging
from typing import List, Tuple, Optional, Union, Any

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

def mask_plate_with_logo(image: np.ndarray, plate_box: Union[List[Tuple[int, int]], Tuple[int, int, int, int]], logo_image: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Overlays the logo on the detected license plate region.
    If logo_image is None or invalid, masks the plate with a black rectangle.

    Args:
        image: The input image
        plate_box: Bounding box coordinates of the license plate.
                  Can be either a list of corner points [(x1,y1), (x2,y2), ...] 
                  or a tuple of (x1, y1, x2, y2)
        logo_image: Logo image to overlay on the plate, or None to use a black rectangle

    Returns:
        Image with the license plate masked
    """
    try:
        # Log input parameters for debugging
        logger.debug(f"[DEBUG] mask_plate_with_logo called with plate_box: {plate_box}")

        # Check if logo_image is None before accessing its shape
        if logo_image is not None:
            try:
                logger.debug(f"[DEBUG] logo_image shape: {logo_image.shape}")
                logger.debug(f"[DEBUG] logo_image type: {type(logo_image)}")
                logger.debug(f"[DEBUG] logo_image dtype: {logo_image.dtype}")
            except Exception as e:
                logger.error(f"[ERROR] Error accessing logo_image properties: {e}")
                logger.error(f"[ERROR] logo_image type: {type(logo_image)}")
                # If logo_image is not a proper numpy array, set it to None
                logo_image = None
        else:
            logger.debug("[DEBUG] logo_image is None")

        # Handle different formats of plate_box
        if isinstance(plate_box, tuple) and len(plate_box) == 4:
            x_min, y_min, x_max, y_max = plate_box
            logger.debug(f"[DEBUG] Plate box is tuple: ({x_min}, {y_min}, {x_max}, {y_max})")
        else:
            try:
                x_coords = [int(point[0]) for point in plate_box]
                y_coords = [int(point[1]) for point in plate_box]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                logger.debug(f"[DEBUG] Plate box converted from points to: ({x_min}, {y_min}, {x_max}, {y_max})")
            except (TypeError, IndexError) as e:
                logger.error(f"[ERROR] Invalid plate_box format: {e}")
                return image

        # Validate plate dimensions
        plate_width = x_max - x_min
        plate_height = y_max - y_min
        logger.debug(f"[DEBUG] Plate dimensions: {plate_width}x{plate_height}")

        if plate_width <= 0 or plate_height <= 0:
            logger.warning(f"[WARNING] Invalid plate dimensions: {plate_width}x{plate_height}, skipping masking")
            return image

        # Validate plate coordinates are within image bounds
        img_height, img_width = image.shape[:2]
        logger.debug(f"[DEBUG] Image dimensions: {img_width}x{img_height}")

        if x_min < 0 or y_min < 0 or x_max > img_width or y_max > img_height:
            logger.warning(f"[WARNING] Plate coordinates ({x_min}, {y_min}, {x_max}, {y_max}) outside image bounds ({img_width}, {img_height}), adjusting")
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            # Recheck dimensions after adjustment
            plate_width = x_max - x_min
            plate_height = y_max - y_min
            logger.debug(f"[DEBUG] Adjusted plate dimensions: {plate_width}x{plate_height}")

            if plate_width <= 0 or plate_height <= 0:
                logger.warning(f"[WARNING] Invalid plate dimensions after adjustment: {plate_width}x{plate_height}, skipping masking")
                return image

        # Extract the region from the original image
        try:
            roi = image[y_min:y_max, x_min:x_max]
            logger.debug(f"[DEBUG] ROI shape: {roi.shape}")
        except Exception as e:
            logger.error(f"[ERROR] Failed to extract ROI: {e}")
            return image

        if logo_image is not None:
            try:
                # Check if logo_image is a valid numpy array with a shape
                if not isinstance(logo_image, np.ndarray):
                    logger.error(f"[ERROR] logo_image is not a numpy array: {type(logo_image)}")
                    # Fall back to black rectangle
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                    logger.info("[INFO] Applied black rectangle mask due to invalid logo_image type.")
                    return image

                # Check if logo_image has a valid shape
                if not hasattr(logo_image, 'shape') or np.prod(logo_image.shape) == 0:
                    logger.error(f"[ERROR] logo_image has invalid shape: {getattr(logo_image, 'shape', 'No shape attribute')}")
                    # Fall back to black rectangle
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                    logger.info("[INFO] Applied black rectangle mask due to invalid logo_image shape.")
                    return image

                # Add alpha channel if missing
                if len(logo_image.shape) == 2:
                    logger.debug("[DEBUG] Converting grayscale logo to BGRA")
                    logo_image = cv2.cvtColor(logo_image, cv2.COLOR_GRAY2BGRA)
                elif logo_image.shape[2] == 3:
                    logger.debug("[DEBUG] Converting BGR logo to BGRA")
                    logo_image = cv2.cvtColor(logo_image, cv2.COLOR_BGR2BGRA)

                logger.debug(f"[DEBUG] Logo shape after conversion: {logo_image.shape}")

                # Resize logo to fit plate region
                try:
                    resized_logo = cv2.resize(logo_image, (plate_width, plate_height))
                    logger.debug(f"[DEBUG] Resized logo shape: {resized_logo.shape}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to resize logo: {e}")
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                    return image

                # Ensure logo has 4 channels (BGRA)
                if resized_logo.shape[2] != 4:
                    logger.warning("[WARNING] Logo does not have alpha channel, creating one")
                    # Create alpha channel (fully opaque)
                    alpha = np.ones((plate_height, plate_width), dtype=np.uint8) * 255
                    if resized_logo.shape[2] == 3:
                        # BGR image
                        resized_logo = cv2.merge([resized_logo[:,:,0], resized_logo[:,:,1], resized_logo[:,:,2], alpha])
                    else:
                        # Grayscale image
                        resized_logo = cv2.merge([resized_logo[:,:,0], resized_logo[:,:,0], resized_logo[:,:,0], alpha])

                    logger.debug(f"[DEBUG] Logo shape after adding alpha: {resized_logo.shape}")

                # Create a mask from the alpha channel of the logo
                try:
                    # Check if resized_logo has at least 4 channels
                    if resized_logo.shape[2] < 4:
                        logger.warning(f"[WARNING] resized_logo has only {resized_logo.shape[2]} channels, expected at least 4")
                        # Add an alpha channel (fully opaque)
                        alpha = np.ones((resized_logo.shape[0], resized_logo.shape[1]), dtype=np.uint8) * 255
                        if resized_logo.shape[2] == 3:
                            # BGR image
                            resized_logo = cv2.merge([resized_logo[:,:,0], resized_logo[:,:,1], resized_logo[:,:,2], alpha])
                        else:
                            # Grayscale image
                            resized_logo = cv2.merge([resized_logo[:,:,0], resized_logo[:,:,0], resized_logo[:,:,0], alpha])
                        logger.debug(f"[DEBUG] Added alpha channel to resized_logo, new shape: {resized_logo.shape}")

                    # Extract alpha channel and adjust to reduce transparency (make more opaque)
                    alpha_channel = resized_logo[:, :, 3] / 255.0
                    # Increase alpha values to reduce transparency (0.7 factor makes it more opaque)
                    alpha_channel = np.clip(alpha_channel * 1.5, 0.0, 1.0)
                    alpha_3channel = np.stack([alpha_channel, alpha_channel, alpha_channel], axis=2)
                    logger.debug(f"[DEBUG] Alpha channel shape: {alpha_3channel.shape}")
                    logger.info("[INFO] Alpha levels adjusted to reduce transparency")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to create alpha channel: {e}")
                    # Fall back to black rectangle
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                    logger.info("[INFO] Applied black rectangle mask due to alpha channel creation failure.")
                    return image

                # Extract BGR channels from the logo
                logo_bgr = resized_logo[:, :, 0:3]
                logger.debug(f"[DEBUG] Logo BGR shape: {logo_bgr.shape}")

                # Ensure roi and logo_bgr have the same shape
                if roi.shape[:2] != logo_bgr.shape[:2]:
                    logger.warning(f"[WARNING] ROI shape {roi.shape[:2]} doesn't match logo shape {logo_bgr.shape[:2]}, resizing logo")
                    try:
                        logo_bgr = cv2.resize(logo_bgr, (roi.shape[1], roi.shape[0]))
                        alpha_3channel = cv2.resize(alpha_3channel, (roi.shape[1], roi.shape[0]))
                        logger.debug(f"[DEBUG] Resized logo BGR shape: {logo_bgr.shape}")
                        logger.debug(f"[DEBUG] Resized alpha channel shape: {alpha_3channel.shape}")
                    except Exception as e:
                        logger.error(f"[ERROR] Failed to resize logo to match ROI: {e}")
                        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                        return image

                # Blend the logo with the original image based on alpha
                try:
                    # Check shapes before blending
                    logger.debug(f"[DEBUG] logo_bgr shape: {logo_bgr.shape}")
                    logger.debug(f"[DEBUG] alpha_3channel shape: {alpha_3channel.shape}")
                    logger.debug(f"[DEBUG] roi shape: {roi.shape}")

                    # Ensure all arrays have the same shape
                    if logo_bgr.shape != roi.shape or logo_bgr.shape[:2] != alpha_3channel.shape[:2]:
                        logger.warning(f"[WARNING] Shape mismatch: logo_bgr={logo_bgr.shape}, roi={roi.shape}, alpha_3channel={alpha_3channel.shape}")

                        # Resize everything to match roi shape
                        logo_bgr = cv2.resize(logo_bgr, (roi.shape[1], roi.shape[0]))
                        alpha_3channel = cv2.resize(alpha_3channel, (roi.shape[1], roi.shape[0]))
                        logger.debug(f"[DEBUG] After resize: logo_bgr={logo_bgr.shape}, alpha_3channel={alpha_3channel.shape}")

                    # Perform the blending
                    blended = (logo_bgr * alpha_3channel) + (roi * (1 - alpha_3channel))
                    blended = blended.astype(np.uint8)
                    logger.debug(f"[DEBUG] Blended image shape: {blended.shape}")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to blend logo with ROI: {e}")
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                    logger.info("[INFO] Applied black rectangle mask due to blending failure.")
                    return image

                # Place the blended image back into the original image
                try:
                    # Check if blended shape matches the ROI shape
                    if blended.shape != roi.shape:
                        logger.warning(f"[WARNING] Blended shape {blended.shape} doesn't match ROI shape {roi.shape}, resizing")
                        blended = cv2.resize(blended, (roi.shape[1], roi.shape[0]))

                    # Check if the coordinates are still valid
                    if y_min < 0 or x_min < 0 or y_max > image.shape[0] or x_max > image.shape[1]:
                        logger.warning(f"[WARNING] Coordinates out of bounds: ({x_min}, {y_min}, {x_max}, {y_max}) for image shape {image.shape}")
                        # Adjust coordinates
                        x_min = max(0, x_min)
                        y_min = max(0, y_min)
                        x_max = min(image.shape[1], x_max)
                        y_max = min(image.shape[0], y_max)

                        # Resize blended to match new dimensions
                        new_width = x_max - x_min
                        new_height = y_max - y_min
                        if new_width > 0 and new_height > 0:
                            blended = cv2.resize(blended, (new_width, new_height))
                        else:
                            logger.error(f"[ERROR] Invalid dimensions after adjustment: {new_width}x{new_height}")
                            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                            return image

                    # Place the blended image
                    image[y_min:y_max, x_min:x_max] = blended
                    logger.info("[SUCCESS] Logo applied to license plate region.")
                except Exception as e:
                    logger.error(f"[ERROR] Failed to place blended image back into original: {e}")
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                    logger.info("[INFO] Applied black rectangle mask due to placement failure.")
            except Exception as e:
                logger.error(f"[ERROR] Failed to apply logo: {e}, using black rectangle instead")
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
        else:
            logger.info(f"[INFO] No valid logo provided (logo_image is {type(logo_image)}), applying black rectangle mask")
            try:
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                logger.info("[SUCCESS] Applied black rectangle mask to license plate region.")
            except Exception as e:
                logger.error(f"[ERROR] Failed to apply black rectangle mask: {e}")
                # Try with adjusted coordinates as a last resort
                try:
                    x_min = max(0, min(x_min, image.shape[1]-1))
                    y_min = max(0, min(y_min, image.shape[0]-1))
                    x_max = max(x_min+1, min(x_max, image.shape[1]))
                    y_max = max(y_min+1, min(y_max, image.shape[0]))
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                    logger.info("[SUCCESS] Applied black rectangle mask with adjusted coordinates.")
                except Exception as e2:
                    logger.error(f"[ERROR] Failed to apply black rectangle mask even with adjusted coordinates: {e2}")
    except Exception as e:
        logger.error(f"[ERROR] Masking failed: {e}")
        # Try to apply a simple black rectangle as a fallback
        try:
            if 'x_min' in locals() and 'y_min' in locals() and 'x_max' in locals() and 'y_max' in locals():
                # Ensure coordinates are valid
                x_min = max(0, min(x_min, image.shape[1]-1))
                y_min = max(0, min(y_min, image.shape[0]-1))
                x_max = max(x_min+1, min(x_max, image.shape[1]))
                y_max = max(y_min+1, min(y_max, image.shape[0]))

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                logger.info("[SUCCESS] Applied fallback black rectangle mask after error.")
            else:
                # If coordinates are not available, try to mask the center of the image
                h, w = image.shape[:2]
                center_x, center_y = w // 2, h // 2
                rect_w, rect_h = w // 4, h // 8  # Reasonable size for a license plate
                x_min, y_min = center_x - rect_w // 2, center_y - rect_h // 2
                x_max, y_max = x_min + rect_w, y_min + rect_h

                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 0), thickness=-1)
                logger.warning("[WARNING] Applied black rectangle to center of image as last resort.")
        except Exception as e2:
            logger.error(f"[ERROR] Even fallback masking failed: {e2}")
            logger.error(f"[ERROR] Image shape: {image.shape if hasattr(image, 'shape') else 'Unknown'}")

    # Final verification that we're returning a valid image
    if not isinstance(image, np.ndarray) or image.size == 0:
        logger.error(f"[ERROR] Invalid image to return: {type(image)}")
        # Create a blank image as a last resort
        return np.zeros((100, 100, 3), dtype=np.uint8)

    logger.debug(f"[DEBUG] Returning image of shape {image.shape}")
    return image
