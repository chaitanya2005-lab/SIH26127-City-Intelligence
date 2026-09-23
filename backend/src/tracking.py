"""
Single-Camera Vehicle Tracking Module (SIH26127)
------------------------------------------------
This module implements persistent single-camera vehicle tracking using Ultralytics YOLO with ByteTrack/BoT-SORT.
It filters specifically for target vehicle classes (car, motorcycle, bus, truck) and assigns persistent
tracking IDs across consecutive video frames.

================================================================================
HOW TO RUN THIS MODULE:
================================================================================
1. Run with a video file (MP4):
   python backend/src/tracking.py --source backend/videos/sample_traffic.mp4

2. Run with webcam input:
   python backend/src/tracking.py --source 0

3. Select tracker algorithm ('bytetrack.yaml' or 'botsort.yaml'):
   python backend/src/tracking.py --source backend/videos/sample.mp4 --tracker bytetrack.yaml
================================================================================
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Union, Optional
import cv2
import numpy as np

# Ensure module directory is in sys.path for direct imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection import VehicleDetector, TARGET_VEHICLE_CLASSES


class SingleCameraTracker(VehicleDetector):
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        tracker_type: str = "bytetrack.yaml",
        target_classes: set = TARGET_VEHICLE_CLASSES,
    ):
        """
        Initialize Single-Camera Vehicle Tracker extending VehicleDetector.

        :param model_path: YOLO model path/identifier.
        :param confidence_threshold: Confidence threshold for vehicle detection.
        :param tracker_type: Tracker configuration file ('bytetrack.yaml' or 'botsort.yaml').
        :param target_classes: Target vehicle classes to track.
        """
        self.tracker_type = tracker_type
        # Initialize base VehicleDetector
        super().__init__(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            target_classes=target_classes,
        )

    def track_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform vehicle detection and assign persistent single-camera tracking IDs for a single frame.

        :param frame: BGR image frame from OpenCV.
        :return: List of tracked objects containing track_id, bbox, class_name, confidence, class_id.
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded.")

        # Run Ultralytics tracking with persistence across consecutive frames
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_type,
            conf=self.confidence_threshold,
            verbose=False,
        )

        tracked_objects = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id].lower()

                # Filter target vehicle classes
                if class_name in self.target_classes:
                    confidence = float(box.conf[0].item())
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    # Extract persistent track ID if assigned by ByteTrack/BoT-SORT
                    track_id = int(box.id[0].item()) if box.id is not None else None

                    tracked_objects.append(
                        {
                            "track_id": track_id,
                            "bbox": xyxy,  # [x1, y1, x2, y2]
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                        }
                    )

        return tracked_objects

    def draw_tracks(self, frame: np.ndarray, tracked_objects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Render bounding boxes, persistent tracking IDs, class names, and confidence scores on frame.

        :param frame: Input BGR image frame.
        :param tracked_objects: List of tracking dictionaries.
        :return: Annotated image frame with tracking visualizations.
        """
        output_frame = frame.copy()

        # Color mapping by vehicle class (BGR)
        colors = {
            "car": (0, 255, 0),          # Green
            "motorcycle": (255, 165, 0),  # Cyan / Blue-Green
            "bus": (0, 165, 255),        # Orange
            "truck": (0, 0, 255),        # Red
        }

        for obj in tracked_objects:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["class_name"].capitalize()
            conf = obj["confidence"]
            track_id = obj["track_id"]
            color = colors.get(obj["class_name"], (255, 255, 255))

            # Bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # Display string including persistent tracking ID: e.g. "ID:3 Car 0.88"
            if track_id is not None:
                display_text = f"ID:{track_id} {label} {conf:.2f}"
            else:
                display_text = f"{label} {conf:.2f}"

            (text_w, text_h), baseline = cv2.getTextSize(
                display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Background rectangle for text label readability
            cv2.rectangle(
                output_frame,
                (x1, max(0, y1 - text_h - 8)),
                (x1 + text_w + 6, max(text_h + 8, y1)),
                color,
                -1,
            )

            # Text color contrast
            text_color = (0, 0, 0) if obj["class_name"] in ["car", "motorcycle"] else (255, 255, 255)
            cv2.putText(
                output_frame,
                display_text,
                (x1 + 3, max(text_h + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
                cv2.LINE_AA,
            )

        return output_frame

    def run_tracking_stream(
        self,
        source: Union[str, int] = 0,
        window_name: str = "SIH26127 - Single-Camera Vehicle Tracking",
    ) -> None:
        """
        Open video stream (MP4 file or webcam index) and display tracked vehicles in real-time.

        :param source: Video file path (str) or webcam index (int/str, e.g. 0).
        :param window_name: Window title.
        """
        if isinstance(source, str) and source.isdigit():
            source = int(source)

        if isinstance(source, int):
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
            if not cap.isOpened() or not cap.grab():
                cap.release()
                cap = cv2.VideoCapture(source)
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[ERROR] Could not open video source: '{source}'. Please verify file path or camera connection.")
            return

        print("[INFO] Single-camera vehicle tracking stream started.")
        print("[INFO] Press 'Q' or 'ESC' while focused on the video display window to quit.")

        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str):
                        print("[INFO] Reached end of video file.")
                    else:
                        print("[WARNING] Failed to grab frame from video source.")
                    break

                # Track objects in frame
                tracked_objects = self.track_frame(frame)

                # Draw track overlays
                annotated_frame = self.draw_tracks(frame, tracked_objects)

                # Calculate real-time FPS
                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-5)
                prev_time = curr_time

                # Display stats header overlay
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f} | Tracked Vehicles: {len(tracked_objects)} | Tracker: {self.tracker_type}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Display frame window
                cv2.imshow(window_name, annotated_frame)

                # Check key press ('q', 'Q', or ESC)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[INFO] Exit key pressed ('Q'). Closing tracking stream.")
                    break

        except KeyboardInterrupt:
            print("[INFO] Tracking stream interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Video resources released and windows closed.")


class MultiCameraTracker:
    def __init__(
        self,
        camera_sources: Dict[str, Union[str, int]],
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        tracker_type: str = "bytetrack.yaml",
    ):
        """
        Initialize Multi-Camera Vehicle Tracker supporting multiple simultaneous camera streams.

        :param camera_sources: Dict mapping camera_id to source (e.g. {'cam_01': 'backend/videos/sample.mp4', 'cam_02': 'backend/videos/sample_cam2.mp4'}).
        :param model_path: YOLO model weights path/identifier.
        :param confidence_threshold: Confidence threshold for vehicle detection.
        :param tracker_type: Tracker configuration file ('bytetrack.yaml' or 'botsort.yaml').
        """
        self.camera_sources = camera_sources
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.tracker_type = tracker_type

        # Initialize independent SingleCameraTracker instance per camera stream to maintain isolated local track IDs
        self.trackers: Dict[str, SingleCameraTracker] = {}
        for camera_id in camera_sources:
            self.trackers[camera_id] = SingleCameraTracker(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                tracker_type=tracker_type,
            )

        # Unified multi-camera spatial-temporal detection log
        self.unified_trajectories: List[Dict[str, Any]] = []

    def record_detection_event(
        self,
        camera_id: str,
        track_id: Optional[int],
        vehicle_class: str,
        confidence: float,
        bbox: List[int],
        timestamp: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store detection event details in the unified multi-camera trajectory log.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        event_record = {
            "camera_id": camera_id,
            "tracking_id": track_id,
            "vehicle_class": vehicle_class,
            "confidence": round(confidence, 3),
            "timestamp": timestamp,
            "bbox": bbox,
        }
        self.unified_trajectories.append(event_record)
        return event_record

    def run_multi_stream(self) -> None:
        """
        Open all camera streams simultaneously, process tracking per frame, store trajectory events,
        and display multi-camera streams in separate windows.
        """
        caps: Dict[str, cv2.VideoCapture] = {}

        # Initialize video captures for each camera source
        for camera_id, source in self.camera_sources.items():
            if isinstance(source, str) and source.isdigit():
                source = int(source)

            if isinstance(source, int):
                cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
                if not cap.isOpened() or not cap.grab():
                    cap.release()
                    cap = cv2.VideoCapture(source)
            else:
                cap = cv2.VideoCapture(source)

            if not cap.isOpened():
                print(f"[ERROR] MultiCameraTracker: Failed to open camera '{camera_id}' source: '{source}'")
            else:
                caps[camera_id] = cap
                print(f"[INFO] Initialized Camera Node '{camera_id}' (Source: {source})")

        if not caps:
            print("[ERROR] No active camera streams could be opened.")
            return

        print(f"[INFO] Multi-Camera vehicle tracking running simultaneously on {len(caps)} feeds.")
        print("[INFO] Press 'Q' or 'ESC' on any display window to exit multi-camera mode.")

        try:
            while True:
                active_streams = 0
                current_timestamp = datetime.now().isoformat()

                for camera_id, cap in list(caps.items()):
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    active_streams += 1
                    tracker_instance = self.trackers[camera_id]

                    # Track frame objects using ByteTrack for this camera feed
                    tracked_objects = tracker_instance.track_frame(frame)

                    # Log detection event into unified multi-camera trajectory log
                    for obj in tracked_objects:
                        self.record_detection_event(
                            camera_id=camera_id,
                            track_id=obj.get("track_id"),
                            vehicle_class=obj.get("class_name"),
                            confidence=obj.get("confidence"),
                            bbox=obj.get("bbox"),
                            timestamp=current_timestamp,
                        )

                    # Draw annotated bounding boxes and track IDs
                    annotated_frame = tracker_instance.draw_tracks(frame, tracked_objects)

                    # Header banner with camera ID
                    cv2.putText(
                        annotated_frame,
                        f"CAM: {camera_id} | Tracked: {len(tracked_objects)}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

                    # Display frame in window named for this camera
                    cv2.imshow(f"SIH26127 Multi-Cam - {camera_id}", annotated_frame)

                if active_streams == 0:
                    print("[INFO] All multi-camera video streams completed.")
                    break

                # Check user exit key press ('q', 'Q', or ESC)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[INFO] Exit key pressed ('Q'). Closing multi-camera streams.")
                    break

        except KeyboardInterrupt:
            print("[INFO] Multi-camera tracking stream interrupted by user.")
        finally:
            for cap in caps.values():
                cap.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Multi-camera streams closed. Total trajectory records logged: {len(self.unified_trajectories)}")


# Keep TrajectoryTracker class stub for multi-camera storage compatibility
class TrajectoryTracker:
    def __init__(self):
        self.trajectories: Dict[str, List[Dict[str, Any]]] = {}

    def log_vehicle_node(self, plate_id: str, camera_id: str, location: Dict[str, float], timestamp: str) -> None:
        if plate_id not in self.trajectories:
            self.trajectories[plate_id] = []
        self.trajectories[plate_id].append({
            "camera_id": camera_id,
            "location": location,
            "timestamp": timestamp
        })

    def get_trajectory(self, plate_id: str) -> List[Dict[str, Any]]:
        return self.trajectories.get(plate_id, [])


def main():
    parser = argparse.ArgumentParser(
        description="SIH26127 - Multi-Camera & Single-Camera Vehicle Tracking Engine."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source for single-camera mode: MP4 video path or camera index (e.g. '0'). Default: '0'.",
    )
    parser.add_argument(
        "--multi-cam",
        action="store_true",
        help="Enable multi-camera test mode running multiple camera sources simultaneously.",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["backend/videos/sample_traffic.mp4", "backend/videos/sample_traffic_cam2.mp4"],
        help="List of camera sources for multi-camera mode. Example: --sources backend/videos/sample_traffic.mp4 backend/videos/sample_traffic_cam2.mp4",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or identifier (e.g. 'yolov8n.pt'). Default: 'yolov8n.pt'.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        choices=["bytetrack.yaml", "botsort.yaml"],
        help="Tracker algorithm config ('bytetrack.yaml' or 'botsort.yaml'). Default: 'bytetrack.yaml'.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for vehicle tracking (0.0 to 1.0). Default: 0.4.",
    )

    args = parser.parse_args()

    if args.multi_cam or len(args.sources) > 1:
        # Build dictionary of camera sources
        camera_map = {}
        for idx, src in enumerate(args.sources, start=1):
            cam_key = f"cam_0{idx}"
            camera_map[cam_key] = src

        print(f"[INFO] Launching Multi-Camera Tracker with feeds: {camera_map}")
        multi_tracker = MultiCameraTracker(
            camera_sources=camera_map,
            model_path=args.model,
            confidence_threshold=args.conf,
            tracker_type=args.tracker,
        )
        multi_tracker.run_multi_stream()
    else:
        # Single-camera mode
        tracker = SingleCameraTracker(
            model_path=args.model,
            confidence_threshold=args.conf,
            tracker_type=args.tracker,
        )
        tracker.run_tracking_stream(source=args.source)


if __name__ == "__main__":
    main()
