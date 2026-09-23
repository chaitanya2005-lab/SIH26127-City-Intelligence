"""
Vehicle Detection Module (SIH26127)
-----------------------------------
This module handles real-time vehicle detection using Ultralytics YOLO models (e.g., YOLOv8n).
It supports video file inputs (e.g., MP4) and live webcam feeds, filtering specifically for target
vehicle classes: car, motorcycle, bus, and truck.

================================================================================
HOW TO RUN THIS MODULE:
================================================================================
1. Ensure dependencies are installed:
   pip install -r backend/requirements.txt

2. Run with default webcam input (webcam 0):
   python backend/src/detection.py --source 0

3. Run with a video file (MP4):
   python backend/src/detection.py --source backend/videos/sample_traffic.mp4

4. Run with a specific YOLO model and confidence threshold:
   python backend/src/detection.py --source 0 --model yolov8n.pt --conf 0.5
================================================================================
"""

import argparse
import sys
import time
from typing import List, Dict, Any, Union
import cv2
import numpy as np

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


# Target vehicle class names to filter for SIH26127
TARGET_VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}


class VehicleDetector:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
        target_classes: set = TARGET_VEHICLE_CLASSES,
    ):
        """
        Initialize the YOLO Vehicle Detector.

        :param model_path: Path or identifier for Ultralytics YOLO model (e.g., 'yolov8n.pt').
        :param confidence_threshold: Minimum confidence threshold for detections.
        :param target_classes: Set of vehicle class names to filter and return.
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_classes = target_classes
        self.model = None
        self.load_model()

    def load_model(self) -> None:
        """
        Load the Ultralytics YOLO model.
        """
        if YOLO is None:
            raise ImportError(
                "Ultralytics package is not installed. Please run 'pip install -r backend/requirements.txt'."
            )
        try:
            print(f"[INFO] Loading YOLO model: '{self.model_path}'...")
            self.model = YOLO(self.model_path)
            print("[INFO] Model loaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from '{self.model_path}': {e}")

    def detect_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Perform vehicle detection on a single frame.

        :param frame: BGR image frame from OpenCV.
        :return: List of detection dictionaries containing bbox, class_name, confidence, class_id.
        """
        if self.model is None:
            raise RuntimeError("YOLO model is not loaded. Call load_model() first.")

        # Run inference with confidence threshold
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id].lower()

                # Filter only requested target vehicle classes
                if class_name in self.target_classes:
                    confidence = float(box.conf[0].item())
                    # Convert bounding box coordinates [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

                    detections.append(
                        {
                            "bbox": xyxy,  # [x1, y1, x2, y2]
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": confidence,
                        }
                    )

        return detections

    def draw_detections(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes, class names, and confidence scores on frame.

        :param frame: Input image frame.
        :param detections: List of detection dictionaries.
        :return: Frame with rendered detection overlays.
        """
        output_frame = frame.copy()

        # Color mapping for vehicle types (BGR)
        colors = {
            "car": (0, 255, 0),          # Green
            "motorcycle": (255, 165, 0),  # Cyan / Blue-Green
            "bus": (0, 165, 255),        # Orange
            "truck": (0, 0, 255),        # Red
        }

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["class_name"].capitalize()
            conf = det["confidence"]
            color = colors.get(det["class_name"], (255, 255, 255))

            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # Prepare text label
            text = f"{label} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Fill background box for text readability
            cv2.rectangle(
                output_frame,
                (x1, max(0, y1 - text_h - 8)),
                (x1 + text_w + 6, max(text_h + 8, y1)),
                color,
                -1,
            )

            # Draw text label in high contrast color
            text_color = (0, 0, 0) if det["class_name"] in ["car", "motorcycle"] else (255, 255, 255)
            cv2.putText(
                output_frame,
                text,
                (x1 + 3, max(text_h + 2, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2,
                cv2.LINE_AA,
            )

        return output_frame

    def run_stream(self, source: Union[str, int] = 0, window_name: str = "SIH26127 - ANPR Vehicle Detection") -> None:
        """
        Open video stream (MP4 file or webcam index) and display detections in real-time.

        :param source: Video file path (str) or webcam index (int/str, e.g. 0).
        :param window_name: Title of the display window.
        """
        # Convert numeric string source to integer (for webcam input like "0")
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
            print(f"[ERROR] Could not open video source: '{source}'. Please verify the file path or webcam connection.")
            return

        print(f"[INFO] Video stream started successfully.")
        print(f"[INFO] Press 'Q' or 'ESC' while focused on the video window to quit.")

        prev_time = time.time()
        fps = 0.0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    if isinstance(source, str):
                        print("[INFO] Reached end of video file.")
                    else:
                        print("[WARNING] Failed to grab frame from video stream.")
                    break

                # Perform vehicle detection
                detections = self.detect_frame(frame)

                # Draw bounding boxes and labels
                annotated_frame = self.draw_detections(frame, detections)

                # Calculate real-time FPS
                curr_time = time.time()
                fps = 1.0 / max(curr_time - prev_time, 1e-5)
                prev_time = curr_time

                # Display stats overlay on top-left
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f} | Vehicles: {len(detections)}",
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Display window
                cv2.imshow(window_name, annotated_frame)

                # Listen for key press ('q', 'Q', or ESC to quit)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):
                    print("[INFO] Exit signal received ('Q'). Closing video stream.")
                    break

        except KeyboardInterrupt:
            print("[INFO] Interrupted by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("[INFO] Video resources released and windows closed.")


def main():
    parser = argparse.ArgumentParser(
        description="SIH26127 - Vehicle Detection Module using Ultralytics YOLO."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Input source: webcam index (e.g., '0') or video file path (e.g., 'backend/videos/sample.mp4'). Default: '0'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model path or identifier (e.g., 'yolov8n.pt'). Default: 'yolov8n.pt'.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.4,
        help="Confidence threshold for vehicle detections (0.0 to 1.0). Default: 0.4.",
    )

    args = parser.parse_args()

    # Initialize VehicleDetector instance
    detector = VehicleDetector(model_path=args.model, confidence_threshold=args.conf)
    
    # Run real-time detection stream
    detector.run_stream(source=args.source)


if __name__ == "__main__":
    main()
