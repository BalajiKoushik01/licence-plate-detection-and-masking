"""
Video-based Vehicle Number Plate Agent
======================================

This module provides an agent for detecting and masking vehicle number plates
in video frames. It extends the functionality of the ImagePlateAgent to video,
ensuring smooth, consistent tracking of vehicles and number plates to maintain
logo placement throughout the clip.
"""

# vehicle_detection/video_plate_agent.py
import os
import cv2
import logging
import numpy as np
import time
from typing import Optional, List, Tuple
from vehicle_detection.image_plate_agent import ImagePlateAgent
from vehicle_detection.detector import detect_vehicles
from vehicle_detection.plate_replacer import overlay_logo_on_frame
logger = logging.getLogger(__name__)

class VideoPlateAgent:
    def __init__(self, ocr_engine: str = "easyocr", conf_threshold: float = 0.25, tracking_method: str = "csrt", existing_reader=None):
        """
        Initialize the Video Plate Agent for license plate detection and processing in videos.

        Args:
            ocr_engine: OCR engine to use for plate recognition.
            conf_threshold: Confidence threshold for vehicle detection.
            tracking_method: Tracking method for video processing.
            existing_reader: Existing OCR reader instance to reuse.
        """
        self.conf_threshold = conf_threshold
        self.ocr_engine = ocr_engine
        self.tracking_method = tracking_method.lower()
        try:
            self.image_agent = ImagePlateAgent(
                ocr_engine=ocr_engine,
                conf_threshold=conf_threshold,
                existing_reader=existing_reader
            )
            logger.info("[SUCCESS] ImagePlateAgent initialized in VideoPlateAgent")
        except Exception as e:
            logger.error(f"[ERROR] Failed to initialize ImagePlateAgent: {e}")
            raise
        self.trackers = {}
        self.plate_boxes = []
        self.vehicle_boxes = []
        self.plate_replacer = None
        self.last_detection_time = 0  # For adaptive detection
        self.min_detection_interval = 15  # Minimum frames between detections
        self.max_detection_interval = 60  # Maximum frames between detections

    def process_video(
        self,
        video_path: str,
        logo_path: Optional[str] = None,
        output_path: Optional[str] = None,
        detection_interval: int = 30,
        show_progress: bool = False,
        use_gpu: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Optional[str]:
        """
        Process a video to detect vehicles, extract license plates, and optionally replace them with a logo.

        Args:
            video_path: Path to the input video.
            logo_path: Path to the logo image for plate replacement (optional).
            output_path: Path to save the processed video (optional).
            detection_interval: Number of frames between detection runs.
            show_progress: Whether to show progress updates.
            use_gpu: Whether to use GPU for processing.
            progress_callback: Callback function to report progress (0-100).

        Returns:
            Path to the processed video, or None if processing fails.
        """
        logger.info(f"[INFO] Processing video: {os.path.basename(video_path)}")

        # Validate input
        if not os.path.exists(video_path):
            logger.error(f"[ERROR] Video not found: {video_path}")
            return None

        # Load logo if provided
        logo = None
        if logo_path:
            logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo is None:
                logger.error(f"[ERROR] Failed to load logo from: {logo_path}")
                return None
            logger.info(f"[INFO] Logo loaded with shape: {logo.shape}")

        # Set up video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"[ERROR] Failed to open video: {video_path}")
            return None

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Fallback to 30 if fps is 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"[INFO] Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")

        # Set up video writer
        final_output_path = output_path or os.path.join(
            "outputs", f"{os.path.splitext(os.path.basename(video_path))[0]}_agent_processed.mp4"
        )
        os.makedirs(os.path.dirname(final_output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(final_output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            logger.error(f"[ERROR] Failed to initialize video writer: {final_output_path}")
            cap.release()
            return None

        frame_count = 0
        last_plate_boxes = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("[INFO] End of video reached")
                break

            frame_count += 1
            if total_frames > 0:
                progress = (frame_count / total_frames) * 100
                if show_progress:
                    logger.info(f"[INFO] Processing frame {frame_count}/{total_frames} ({progress:.1f}%)")

                # Call progress callback if provided
                if progress_callback:
                    try:
                        progress_callback(progress)
                    except Exception as e:
                        logger.error(f"[ERROR] Progress callback failed: {e}")

            try:
                # Update existing trackers
                tracked_boxes = self._update_trackers(frame)

                # Detect new plates at specified intervals
                current_interval = self._calculate_adaptive_interval(frame_count, tracked_boxes)
                if frame_count % current_interval == 0:
                    new_plate_boxes = self._detect_new_plates(frame, frame_count, current_interval)
                    if len(new_plate_boxes) > 0:
                        last_plate_boxes = new_plate_boxes
                        self._update_plate_boxes(new_plate_boxes)
                    else:
                        last_plate_boxes = tracked_boxes if len(tracked_boxes) > 0 else last_plate_boxes
                else:
                    last_plate_boxes = tracked_boxes if len(tracked_boxes) > 0 else last_plate_boxes

                # Process the frame with logo overlay or masking
                if logo is not None and len(last_plate_boxes) > 0:
                    formatted_boxes = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)] for x1, y1, x2, y2 in last_plate_boxes]
                    processed_frame = overlay_logo_on_frame(frame, formatted_boxes, logo)
                else:
                    processed_frame = frame

                out.write(processed_frame)

            except Exception as e:
                logger.error(f"[ERROR] Error processing frame {frame_count}: {e}")
                out.write(frame)  # Write original frame to avoid breaking the video
                continue

        cap.release()
        out.release()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"[WARNING] Error closing OpenCV windows: {e}")

        if os.path.exists(final_output_path) and os.path.getsize(final_output_path) > 0:
            logger.info(f"[SUCCESS] Video processed. Output saved to: {final_output_path}")
            return final_output_path
        else:
            logger.error(f"[ERROR] Failed to save video: {final_output_path}")
            if os.path.exists(final_output_path):
                os.remove(final_output_path)
            return None

    def _create_tracker(self):
        """
        Create a tracker based on the specified tracking method.

        Returns:
            OpenCV tracker object
        """
        if self.tracking_method == 'csrt':
            return cv2.legacy.TrackerCSRT_create()
        elif self.tracking_method == 'kcf':
            return cv2.legacy.TrackerKCF_create()
        elif self.tracking_method == 'boosting':
            return cv2.legacy.TrackerBoosting_create()
        elif self.tracking_method == 'mil':
            return cv2.legacy.TrackerMIL_create()
        elif self.tracking_method == 'tld':
            return cv2.legacy.TrackerTLD_create()
        elif self.tracking_method == 'medianflow':
            return cv2.legacy.TrackerMedianFlow_create()
        elif self.tracking_method == 'mosse':
            return cv2.legacy.TrackerMOSSE_create()
        else:
            logger.warning(f"Unknown tracking method: {self.tracking_method}, using CSRT instead")
            return cv2.legacy.TrackerCSRT_create()

    def _initialize_trackers(self, frame, plate_boxes):
        """
        Initialize trackers for each plate in the frame.

        Args:
            frame: Current video frame
            plate_boxes: List of plate bounding boxes

        Returns:
            Dictionary of initialized trackers
        """
        trackers = {}
        for i, plate_box in enumerate(plate_boxes):
            if isinstance(plate_box, tuple) and len(plate_box) == 4:
                x_min, y_min, x_max, y_max = plate_box
            else:
                # Convert polygon points to bounding box
                x_coords = [int(point[0]) for point in plate_box]
                y_coords = [int(point[1]) for point in plate_box]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)

            # Create and initialize tracker
            tracker = self._create_tracker()
            bbox = (x_min, y_min, x_max - x_min, y_max - y_min)
            success = tracker.init(frame, bbox)

            if success:
                trackers[i] = {
                    'tracker': tracker,
                    'bbox': bbox,
                    'original_box': plate_box,
                    'frames_since_detection': 0,
                    'confidence': 1.0
                }
            else:
                logger.warning(f"Failed to initialize tracker for plate {i}")

        return trackers

    def _update_trackers(self, frame):
        """
        Update all trackers with the current frame.

        Args:
            frame: Current video frame

        Returns:
            Updated list of plate boxes
        """
        updated_boxes = []
        trackers_to_remove = []

        for plate_id, tracker_info in self.trackers.items():
            tracker = tracker_info['tracker']
            success, bbox = tracker.update(frame)

            if success:
                x, y, w, h = map(int, bbox)
                updated_box = (x, y, x + w, y + h)

                # Update tracker info
                tracker_info['bbox'] = bbox
                tracker_info['frames_since_detection'] += 1

                # Decrease confidence over time
                tracker_info['confidence'] *= 0.98  # Adjusted decay rate

                # If confidence is too low, mark for removal
                if tracker_info['confidence'] < 0.4 or tracker_info['frames_since_detection'] > 150:  # Added max age
                    trackers_to_remove.append(plate_id)
                else:
                    updated_boxes.append(updated_box)
            else:
                trackers_to_remove.append(plate_id)

        # Remove failed trackers
        for plate_id in trackers_to_remove:
            logger.debug(f"[DEBUG] Removing tracker {plate_id} due to low confidence or tracking failure")
            del self.trackers[plate_id]

        return updated_boxes

    def _detect_vehicles(self, frame):
        """
        Detect vehicles in the given frame.

        Args:
            frame: Current video frame
        Returns:
            List of detected vehicle bounding boxes
        """
        vehicle_boxes = detect_vehicles(frame, self.conf_threshold)
        return vehicle_boxes

    def _detect_new_plates(self, frame, frame_count, detection_interval=30):
        """
        Detect new plates in the frame at regular intervals.

        Args:
            frame: Current video frame
            frame_count: Current frame number
            detection_interval: Interval for running detection

        Returns:
            List of newly detected plate boxes
        """
        if frame_count % detection_interval != 0:
            return self.plate_boxes

        start_time = time.time()
        _, vehicle_boxes, plate_boxes = self.image_agent.detect_plates(frame)
        detection_time = time.time() - start_time
        logger.debug(f"[DEBUG] Plate detection took {detection_time:.2f} seconds")

        # Update vehicle and plate boxes
        self._update_vehicle_boxes(vehicle_boxes)
        new_trackers = self._initialize_trackers(frame, plate_boxes)

        # Merge with existing trackers, avoiding duplicates
        for plate_id, tracker_info in new_trackers.items():
            is_duplicate = False
            for existing_id, existing_info in self.trackers.items():
                existing_bbox = existing_info['bbox']
                new_bbox = tracker_info['bbox']

                # Calculate IoU (Intersection over Union)
                x1 = max(existing_bbox[0], new_bbox[0])
                y1 = max(existing_bbox[1], new_bbox[1])
                x2 = min(existing_bbox[0] + existing_bbox[2], new_bbox[0] + new_bbox[2])
                y2 = min(existing_bbox[1] + existing_bbox[3], new_bbox[1] + new_bbox[3])

                if x2 > x1 and y2 > y1:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = existing_bbox[2] * existing_bbox[3]
                    area2 = new_bbox[2] * new_bbox[3]
                    union = area1 + area2 - intersection
                    iou = intersection / union if union > 0 else 0

                    if iou > 0.5:
                        is_duplicate = True
                        existing_info['frames_since_detection'] = 0
                        existing_info['confidence'] = 1.0
                        break

            if not is_duplicate:
                new_id = max(self.trackers.keys()) + 1 if self.trackers else 0
                self.trackers[new_id] = tracker_info

        return plate_boxes

    def _update_plate_boxes(self, plate_boxes):
        """
        Update the list of plate boxes with the current frame's plate boxes.

        Args:
            plate_boxes: List of detected plate bounding boxes
        """
        self.plate_boxes = plate_boxes

    def _update_vehicle_boxes(self, vehicle_boxes):
        """
        Update the list of vehicle boxes with the current frame's vehicle boxes.

        Args:
            vehicle_boxes: List of detected vehicle bounding boxes
        """
        self.vehicle_boxes = vehicle_boxes

    def _calculate_adaptive_interval(self, frame_count: int, tracked_boxes: List[Tuple[int, int, int, int]]) -> int:
        """
        Calculate an adaptive detection interval based on tracking performance.

        Args:
            frame_count: Current frame number
            tracked_boxes: Currently tracked plate boxes

        Returns:
            Adjusted detection interval
        """
        current_time = time.time()
        time_since_last_detection = current_time - self.last_detection_time

        # If no plates are being tracked, reduce interval to detect sooner
        if not tracked_boxes:
            interval = self.min_detection_interval
        else:
            # If trackers are stable, increase interval
            avg_confidence = sum(info['confidence'] for info in self.trackers.values()) / len(self.trackers) if self.trackers else 0
            if avg_confidence > 0.8:
                interval = min(self.max_detection_interval, int(self.detection_interval * 1.5))
            else:
                interval = max(self.min_detection_interval, int(self.detection_interval * 0.5))

        # Ensure at least min interval has passed since last detection
        if time_since_last_detection * 30 < interval:  # Assuming 30 FPS
            interval = max(self.min_detection_interval, int(time_since_last_detection * 30))

        if frame_count % interval == 0:
            self.last_detection_time = current_time

        return interval

    def mask_plate_with_logo(self, frame: np.ndarray, plate_box: Tuple[int, int, int, int], logo_path: Optional[str] = None) -> np.ndarray:
        """
        Mask the detected plate with a logo.

        Args:
            frame: Current video frame
            plate_box: Bounding box of the detected plate
            logo_path: Path to the logo image

        Returns:
            Frame with the logo applied to the plate
        """
        if logo_path and os.path.exists(logo_path):
            return overlay_logo_on_frame(frame, [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)] for x1, y1, x2, y2 in [plate_box]], cv2.imread(logo_path, cv2.IMREAD_UNCHANGED))
        else:
            logger.warning(f"Logo path not found or invalid: {logo_path}, applying simple mask")
            x1, y1, x2, y2 = plate_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)  # Black rectangle as fallback
            return frame

    def detect_plates(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect plates in the given frame.

        Args:
            frame: Current video frame
        Returns:
            List of detected plate bounding boxes
        """
        _, vehicle_boxes, plate_boxes = self.image_agent.detect_plates(frame)
        return plate_boxes

    def process_video_with_logo(
        self, 
        video_path: str, 
        logo_path: Optional[str] = None,
        use_gpu: bool = False,
        progress_callback: Optional[callable] = None
    ) -> Optional[str]:
        """
        Process a video to detect vehicles, extract license plates, and replace them with a logo.

        Args:
            video_path: Path to the input video.
            logo_path: Path to the logo image for plate replacement (optional).
            use_gpu: Whether to use GPU for processing.
            progress_callback: Callback function to report progress (0-100).

        Returns:
            Path to the processed video, or None if processing fails.
        """
        return self.process_video(
            video_path, 
            logo_path=logo_path, 
            output_path=None,
            use_gpu=use_gpu,
            progress_callback=progress_callback
        )

    def process_realtime(
        self,
        camera_id: int = 0,
        logo_path: Optional[str] = None,
        output_path: Optional[str] = None,
        detection_interval: int = 30,
        show_display: bool = True
    ) -> None:
        """
        Process a real-time video stream from a camera.

        Args:
            camera_id: Camera ID to use
            logo_path: Path to the logo image
            output_path: Path to save the output video
            detection_interval: Interval for running detection (in frames)
            show_display: Whether to show the processed video in a window
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            logger.error(f"[ERROR] Failed to open camera with ID {camera_id}")
            raise ValueError(f"Failed to open camera with ID {camera_id}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        logger.info(f"[INFO] Camera properties: {width}x{height}, {fps} FPS")

        # Load logo if provided
        logo = None
        if logo_path and os.path.exists(logo_path):
            logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
            if logo is None:
                logger.error(f"[ERROR] Failed to load logo from: {logo_path}")
                logo = None
            else:
                logger.info(f"[INFO] Logo loaded with shape: {logo.shape}")

        # Create VideoWriter if output_path is provided
        out = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not out.isOpened():
                logger.error(f"[ERROR] Failed to initialize video writer: {output_path}")
                cap.release()
                return

        # Reset trackers
        self.trackers = {}

        # Process each frame
        frame_count = 0
        logger.info("Starting real-time processing. Press 'q' to quit.")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("[INFO] End of camera stream")
                break

            start_time = time.time()

            # Update existing trackers
            tracked_boxes = self._update_trackers(frame)

            # Detect new plates at adaptive intervals
            current_interval = self._calculate_adaptive_interval(frame_count, tracked_boxes)
            if frame_count % current_interval == 0:
                new_plate_boxes = self._detect_new_plates(frame, frame_count, current_interval)
                plate_boxes_to_use = new_plate_boxes if len(new_plate_boxes) > 0 else tracked_boxes
            else:
                plate_boxes_to_use = tracked_boxes

            # Process each tracked plate
            result_frame = frame.copy()
            if len(plate_boxes_to_use) > 0:
                if logo:
                    formatted_boxes = [[(x1, y1), (x2, y1), (x2, y2), (x1, y2)] for x1, y1, x2, y2 in plate_boxes_to_use]
                    result_frame = overlay_logo_on_frame(result_frame, formatted_boxes, logo)
                else:
                    for plate_box in plate_boxes_to_use:
                        result_frame = self.mask_plate_with_logo(result_frame, plate_box, None)

            # Calculate FPS
            process_time = time.time() - start_time
            fps_text = f"FPS: {1/process_time:.1f}" if process_time > 0 else "FPS: N/A"

            # Add FPS text to frame
            cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Write processed frame if output_path is provided
            if out:
                out.write(result_frame)

            # Show frame if requested
            if show_display:
                cv2.imshow('Real-time Processing', result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_count += 1

        # Release resources
        cap.release()
        if out:
            out.release()
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.warning(f"[WARNING] Error closing OpenCV windows: {e}")
        logger.info("Real-time processing stopped")
