"""
Video Processing Module
======================

This module provides functionality for processing videos for vehicle detection,
license plate recognition, and replacement with photorealistic logo overlays.
"""

import cv2
import os
import time
import logging
import subprocess
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import torch
import ffmpeg

# Import detection and processing functions
from vehicle_detection.detector import detect_vehicles, model, VEHICLE_CLASSES, DEFAULT_COLORS, VehicleDetector
from vehicle_detection.plate_recognizer import extract_license_plates
from vehicle_detection.plate_replacer import PlateReplacer

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

class VideoProcessor:
    """
    Class for processing videos with vehicle and license plate detection and replacement.

    Features:
    - Detects vehicles and license plates in videos using YOLOv8 and OCR.
    - Tracks license plates across frames using IoU-based tracking.
    - Replaces license plates with photorealistic logos.
    - Supports real-time webcam processing and video file processing.
    """

    def __init__(
        self,
        conf_threshold: float = 0.25,
        ocr_engine: str = "easyocr",
        use_gpu: bool = False,
        plate_model_path: str = "yolov8_plate.pt",
        region: str = "India"
    ):
        """
        Initialize the VideoProcessor.

        Args:
            conf_threshold: Confidence threshold for vehicle detection.
            ocr_engine: OCR engine to use for license plate recognition ("easyocr" or "tesseract").
            use_gpu: Whether to use GPU for processing.
            plate_model_path: Path to the fine-tuned YOLOv8 model for plate detection.
            region: Region for plate text validation (e.g., "India", "US", "EU").
        """
        self.conf_threshold = conf_threshold
        self.ocr_engine = ocr_engine
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.plate_model_path = plate_model_path
        self.region = region
        self.vehicle_detector = VehicleDetector(use_gpu=self.use_gpu)
        self.plate_recognizer = extract_license_plates
        self.plate_replacer = PlateReplacer(use_gpu=self.use_gpu)
        self.tracked_plates = {}  # Dictionary to store tracked plates
        self.last_detection_time = 0
        self.min_detection_interval = 15
        self.max_detection_interval = 60
        self.frame_count = 0
        logger.info(f"[INFO] Initialized VideoProcessor with {ocr_engine} OCR engine (GPU: {self.use_gpu})")

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes.

        Args:
            box1: First bounding box (x1, y1, x2, y2).
            box2: Second bounding box (x1, y1, x2, y2).

        Returns:
            IoU value as a float.
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1 = max(x1_1, x1_2)
        y1 = max(y1_1, y1_2)
        x2 = min(x2_1, x2_2)
        y2 = min(y2_1, y2_2)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def _update_tracked_plates(self, new_plates: List[Tuple[str, Tuple[int, int, int, int]]]) -> List[Tuple[str, Tuple[int, int, int, int]]]:
        """
        Update tracked plates using IoU-based matching and temporal smoothing.

        Args:
            new_plates: List of newly detected plates with text and bounding boxes.

        Returns:
            Updated list of tracked plates.
        """
        current_plates = []
        new_tracked_plates = {}

        # Match new plates with existing tracked plates using IoU
        for plate_id, plate_info in self.tracked_plates.items():
            best_iou = 0
            best_new_plate = None
            old_box = plate_info['box']

            for text, new_box in new_plates:
                iou = self._calculate_iou(old_box, new_box)
                if iou > best_iou and iou > 0.5:
                    best_iou = iou
                    best_new_plate = (text, new_box)

            if best_new_plate:
                # Smooth the bounding box (exponential moving average)
                alpha = 0.7
                new_box = best_new_plate[1]
                smoothed_box = (
                    int(alpha * old_box[0] + (1 - alpha) * new_box[0]),
                    int(alpha * old_box[1] + (1 - alpha) * new_box[1]),
                    int(alpha * old_box[2] + (1 - alpha) * new_box[2]),
                    int(alpha * old_box[3] + (1 - alpha) * new_box[3])
                )
                new_tracked_plates[plate_id] = {
                    'text': best_new_plate[0],
                    'box': smoothed_box,
                    'frames_since_detection': 0,
                    'confidence': 1.0
                }
                current_plates.append((best_new_plate[0], smoothed_box))
            else:
                # No match found, increment age
                plate_info['frames_since_detection'] += 1
                plate_info['confidence'] *= 0.98
                if plate_info['frames_since_detection'] <= 150 and plate_info['confidence'] > 0.4:
                    new_tracked_plates[plate_id] = plate_info
                    current_plates.append((plate_info['text'], plate_info['box']))

        # Add unmatched new plates as new tracked objects
        matched_ids = set()
        for plate_id in new_tracked_plates:
            for text, new_box in new_plates:
                if self._calculate_iou(new_tracked_plates[plate_id]['box'], new_box) > 0.5:
                    matched_ids.add(id(new_box))

        next_id = max(self.tracked_plates.keys(), default=-1) + 1
        for text, new_box in new_plates:
            if id(new_box) not in matched_ids:
                new_tracked_plates[next_id] = {
                    'text': text,
                    'box': new_box,
                    'frames_since_detection': 0,
                    'confidence': 1.0
                }
                current_plates.append((text, new_box))
                next_id += 1

        self.tracked_plates = new_tracked_plates
        logger.debug(f"[DEBUG] Updated {len(current_plates)} tracked plates")
        return current_plates

    def _detect_vehicles_and_plates(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[str, Tuple[int, int, int, int]]]]:
        """
        Detect vehicles and license plates in the frame.

        Args:
            frame: Current video frame.

        Returns:
            Tuple of (vehicle_boxes, plate_boxes_with_text).
        """
        if frame is None or frame.size == 0:
            logger.error("[ERROR] Invalid frame provided to _detect_vehicles_and_plates")
            return [], []

        # Detect vehicles
        vehicle_boxes = self.vehicle_detector.detect(frame, conf_threshold=self.conf_threshold)
        vehicle_boxes = [(int(x1), int(y1), int(x2), int(y2)) for x1, y1, x2, y2 in vehicle_boxes]

        # Validate vehicle boxes
        vehicle_boxes = [
            box for box in vehicle_boxes
            if box[0] < box[2] and box[1] < box[3] and
               box[0] >= 0 and box[1] >= 0 and
               box[2] <= frame.shape[1] and box[3] <= frame.shape[0]
        ]

        # Detect license plates
        plates = extract_license_plates(
            image_path=frame,
            car_boxes=vehicle_boxes,
            ocr_engine=self.ocr_engine,
            conf_threshold=self.conf_threshold,
            use_gpu=self.use_gpu,
            plate_model_path=self.plate_model_path,
            region=self.region
        )

        logger.info(f"[INFO] Detected {len(vehicle_boxes)} vehicles and {len(plates)} plates")
        return vehicle_boxes, plates

    def _calculate_adaptive_interval(self, tracked_plates: List[Tuple[str, Tuple[int, int, int, int]]], base_interval: int) -> int:
        """
        Calculate an adaptive detection interval based on tracking performance and frame content.

        Args:
            tracked_plates: Currently tracked plates.
            base_interval: Base detection interval.

        Returns:
            Adjusted detection interval.
        """
        current_time = time.time()
        time_since_last_detection = current_time - self.last_detection_time

        if not tracked_plates:
            interval = self.min_detection_interval
        else:
            avg_confidence = sum(info['confidence'] for info in self.tracked_plates.values()) / len(self.tracked_plates) if self.tracked_plates else 0
            num_plates = len(tracked_plates)
            if avg_confidence > 0.8 and num_plates < 3:
                interval = min(self.max_detection_interval, int(base_interval * 1.5))
            elif num_plates > 5:
                interval = max(self.min_detection_interval, int(base_interval * 0.5))
            else:
                interval = base_interval

        if time_since_last_detection * 30 < interval:
            interval = max(self.min_detection_interval, int(time_since_last_detection * 30))

        return interval

    def process_frame(
        self,
        frame: np.ndarray,
        logo_path: Optional[str] = None,
        detection_interval: int = 30
    ) -> np.ndarray:
        """
        Process a single frame of the video.

        Args:
            frame: Current video frame.
            logo_path: Path to the logo image.
            detection_interval: Interval for running detection.

        Returns:
            Processed frame with vehicle detections and plate replacements.
        """
        if frame is None or frame.size == 0:
            logger.error("[ERROR] Invalid frame provided to process_frame")
            return frame

        self.frame_count += 1

        # Detect vehicles and plates at adaptive intervals
        current_interval = self._calculate_adaptive_interval(
            [(text, box) for text, box in self.tracked_plates.values()],
            detection_interval
        )
        if self.frame_count % current_interval == 0:
            vehicle_boxes, new_plates = self._detect_vehicles_and_plates(frame)
            self.last_detection_time = time.time()
            current_plates = self._update_tracked_plates(new_plates)
        else:
            current_plates = [(info['text'], info['box']) for info in self.tracked_plates.values()]

        # Overlay logos on plates
        plate_boxes = [box for _, box in current_plates]
        result_frame = self.plate_replacer.overlay_logo_on_frame(
            frame=frame,
            plate_boxes=plate_boxes,
            logo=cv2.imread(logo_path, cv2.IMREAD_UNCHANGED) if logo_path and os.path.exists(logo_path) else None
        )

        # Draw vehicle bounding boxes
        for box in vehicle_boxes:
            x1, y1, x2, y2 = box
            color = DEFAULT_COLORS.get("Vehicle", (0, 255, 0))
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
            label = "Vehicle"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_frame, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(result_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return result_frame

def extract_frames_with_ffmpeg(
    video_path: str,
    frames_dir: str,
    fps: int = 30,
    start_time: Optional[str] = None,
    duration: Optional[str] = None
) -> bool:
    """
    Extract frames from video using FFmpeg.

    Args:
        video_path: Path to the input video file.
        frames_dir: Directory to save extracted frames.
        fps: Frames per second to extract.
        start_time: Optional start time in format "HH:MM:SS".
        duration: Optional duration in format "HH:MM:SS".

    Returns:
        True if successful, False otherwise.
    """
    try:
        os.makedirs(frames_dir, exist_ok=True)
        stream = ffmpeg.input(video_path, ss=start_time, t=duration)
        stream = ffmpeg.output(
            stream,
            os.path.join(frames_dir, "frame_%04d.jpg"),
            vf=f"fps={fps}",
            q_v=2,
            loglevel="quiet"
        )
        ffmpeg.run(stream)
        logger.info(f"[SUCCESS] Frames extracted to {frames_dir}")
        return True
    except ffmpeg.Error as e:
        logger.error(f"[ERROR] FFmpeg extraction failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Frame extraction failed: {e}")
        return False

def create_video_with_ffmpeg(
    frames_dir: str,
    output_video_path: str,
    fps: int = 30,
    codec: str = "libx264"
) -> bool:
    """
    Create video from frames using FFmpeg.

    Args:
        frames_dir: Directory containing frames.
        output_video_path: Path to save the output video.
        fps: Frames per second for the output video.
        codec: Video codec to use.

    Returns:
        True if successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        stream = ffmpeg.input(
            os.path.join(frames_dir, "frame_%04d.jpg"),
            framerate=fps
        )
        stream = ffmpeg.output(
            stream,
            output_video_path,
            c_v=codec,
            pix_fmt="yuv420p",
            crf=23,
            loglevel="quiet"
        )
        ffmpeg.run(stream)
        logger.info(f"[SUCCESS] Video created: {output_video_path}")
        return True
    except ffmpeg.Error as e:
        logger.error(f"[ERROR] FFmpeg video creation failed: {e.stderr.decode()}")
        return False
    except Exception as e:
        logger.error(f"[ERROR] Video creation failed: {e}")
        return False

def real_time_detection(
    conf_threshold: float = 0.25,
    custom_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    camera_id: int = 0,
    save_output: bool = False,
    output_path: Optional[str] = None,
    logo_path: Optional[str] = None,
    use_gpu: bool = False,
    plate_model_path: str = "yolov8_plate.pt",
    region: str = "India"
) -> None:
    """
    Perform real-time vehicle detection and plate replacement using a webcam.

    Args:
        conf_threshold: Confidence threshold for detections.
        custom_colors: Custom colors for different vehicle types.
        camera_id: Camera device ID (usually 0 for built-in webcam).
        save_output: Whether to save the output video.
        output_path: Path to save the output video (if save_output is True).
        logo_path: Path to the logo image for plate replacement.
        use_gpu: Whether to use GPU for processing.
        plate_model_path: Path to the fine-tuned YOLOv8 model for plate detection.
        region: Region for plate text validation (e.g., "India", "US", "EU").
    """
    processor = VideoProcessor(
        conf_threshold=conf_threshold,
        use_gpu=use_gpu,
        plate_model_path=plate_model_path,
        region=region
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        logger.error(f"[ERROR] Unable to access the webcam with ID {camera_id}.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = None
    if save_output and output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        logger.info(f"[INFO] Saving output to: {output_path}")

    logger.info("[INFO] Starting real-time vehicle detection and plate replacement. Press 'q' to exit.")
    frame_count = 0
    frame_skip = 0
    max_frame_skip = 2  # Skip frames if processing is too slow

    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("[ERROR] Failed to capture frame from webcam.")
            break

        frame_count += 1
        start_time = time.time()

        # Skip frames if processing is too slow
        if frame_skip > 0:
            frame_skip -= 1
            continue

        try:
            processed_frame = processor.process_frame(frame, logo_path, detection_interval=15)
        except Exception as e:
            logger.error(f"[ERROR] Failed to process frame {frame_count}: {e}")
            processed_frame = frame

        # Calculate FPS and adjust frame skipping
        process_time = time.time() - start_time
        actual_fps = 1 / process_time if process_time > 0 else 0
        if actual_fps < fps * 0.8 and frame_skip < max_frame_skip:
            frame_skip = min(max_frame_skip, int((fps / actual_fps) - 1))
            logger.debug(f"[DEBUG] Frame skipping enabled: {frame_skip} frames")

        fps_text = f"FPS: {actual_fps:.1f}"
        cv2.putText(processed_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if save_output and out is not None:
            out.write(processed_frame)

        cv2.imshow("Real-Time Vehicle Detection and Plate Replacement", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if save_output and out is not None:
        out.release()
    cv2.destroyAllWindows()
    logger.info("[INFO] Real-time detection stopped.")

def process_video(
    video_path: str,
    logo_path: Optional[str] = None,
    output_path: Optional[str] = None,
    detection_interval: int = 30,
    show_progress: bool = False,
    conf_threshold: float = 0.25,
    use_gpu: bool = False,
    plate_model_path: str = "yolov8_plate.pt",
    region: str = "India",
    detect_plates: bool = True,
    replace_plates: bool = True,
    ocr_engine: str = "easyocr",
    show: bool = False,
    progress_callback: Optional[callable] = None
) -> Optional[str]:
    """
    Process a video to detect vehicles, extract license plates, and optionally replace them with a logo.

    Args:
        video_path: Path to the input video file.
        logo_path: Path to the logo image.
        output_path: Path to save the output video.
        detection_interval: Interval for running detection.
        show_progress: Whether to show progress during processing.
        conf_threshold: Confidence threshold for vehicle detection.
        use_gpu: Whether to use GPU for processing.
        plate_model_path: Path to the fine-tuned YOLOv8 model for plate detection.
        region: Region for plate text validation (e.g., "India", "US", "EU").

    Returns:
        Path to the output video file.
    """
    logger.info(f"[INFO] Processing video: {os.path.basename(video_path)}")

    if not os.path.exists(video_path):
        logger.error(f"[ERROR] Video not found: {video_path}")
        return None

    processor = VideoProcessor(
        conf_threshold=conf_threshold,
        use_gpu=use_gpu,
        plate_model_path=plate_model_path,
        region=region
    )

    try:
        probe = ffmpeg.probe(video_path)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        width = int(video_info['width'])
        height = int(video_info['height'])
        fps = float(video_info.get('r_frame_rate', '30/1').split('/')[0])
        total_frames = int(video_info.get('nb_frames', 0))
    except ffmpeg.Error as e:
        logger.error(f"[ERROR] Failed to probe video: {e.stderr.decode()}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[ERROR] Failed to open video: {video_path}")
        return None

    final_output_path = output_path or os.path.join(
        "outputs", f"{os.path.splitext(os.path.basename(video_path))[0]}_processed.mp4"
    )
    os.makedirs(os.path.dirname(final_output_path), exist_ok=True)

    # Use FFmpeg for video writing
    try:
        process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f'{width}x{height}', framerate=fps)
            .output(final_output_path, pix_fmt='yuv420p', vcodec='libx264', crf=23, loglevel="quiet")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
    except ffmpeg.Error as e:
        logger.error(f"[ERROR] Failed to initialize FFmpeg video writer: {e.stderr.decode()}")
        cap.release()
        return None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("[INFO] End of video reached")
            break

        frame_count += 1
        if show_progress and total_frames > 0:
            progress = (frame_count / total_frames) * 100
            logger.info(f"[INFO] Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")

            # Call progress callback if provided
            if progress_callback:
                try:
                    progress_callback(progress)
                except Exception as e:
                    logger.error(f"[ERROR] Progress callback failed: {e}")

        try:
            # Only use logo_path if replace_plates is True
            logo_to_use = logo_path if replace_plates else None
            processed_frame = processor.process_frame(frame, logo_to_use, detection_interval)

            # Show frame if requested
            if show:
                cv2.imshow("Processing Video", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            process.stdin.write(processed_frame.tobytes())
        except Exception as e:
            logger.error(f"[ERROR] Error processing frame {frame_count}: {e}")
            process.stdin.write(frame.tobytes())
            continue

    cap.release()
    try:
        process.stdin.close()
        process.wait()
    except ffmpeg.Error as e:
        logger.error(f"[ERROR] FFmpeg video writing failed: {e.stderr.decode()}")

    if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
        logger.info(f"[SUCCESS] Video processed. Output saved to: {final_output_path}")
        return final_output_path
    else:
        logger.error(f"[ERROR] Failed to save video: {final_output_path}")
        if os.path.exists(final_output_path):
            os.remove(final_output_path)
        return None

if __name__ == "__main__":
    print(f"CUDA Available: {torch.cuda.is_available()}")
    video_path = "path/to/video.mp4"
    logo_path = "path/to/logo.png"
    output_path = "path/to/output_video.mp4"
    process_video(video_path, logo_path, output_path, show_progress=True)
