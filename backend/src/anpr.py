"""
Automatic Number Plate Recognition (ANPR) Module (SIH26127)
-------------------------------------------------------------
Detects license plate regions within vehicle bounding boxes and extracts
license plate alphanumeric text using EasyOCR / OpenCV with regex normalization.
Associates recognized plates with vehicle tracking IDs and camera IDs.

================================================================================
HOW TO RUN THIS MODULE:
================================================================================
1. Run ANPR on single video feed / webcam:
   python backend/src/anpr.py --source backend/videos/sample_traffic.mp4

2. Run ANPR on live webcam:
   python backend/src/anpr.py --source 0

3. Run ANPR in multi-camera test mode:
   python backend/src/anpr.py --multi-cam
================================================================================
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Union
import cv2
import numpy as np

# Ensure module directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detection import VehicleDetector, TARGET_VEHICLE_CLASSES
from tracking import SingleCameraTracker, MultiCameraTracker

try:
    import easyocr
except ImportError:
    easyocr = None


class ANPREngine:
    def __init__(
        self,
        ocr_languages: Optional[List[str]] = None,
        gpu: bool = False,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize ANPR Engine with EasyOCR reader and plate text normalization rules.

        :param ocr_languages: List of language codes for OCR reader (default: ['en']).
        :param gpu: Whether to enable CUDA GPU acceleration if available.
        :param confidence_threshold: Minimum confidence score to accept an OCR text result.
        """
        self.languages = ocr_languages or ["en"]
        self.gpu = gpu
        self.confidence_threshold = confidence_threshold
        self.reader = None
        self.initialize_ocr()

    def initialize_ocr(self) -> None:
        """
        Initialize EasyOCR reader into memory safely without crashing.
        """
        if easyocr is None:
            print("[INFO] EasyOCR package not detected. Running ANPR in contour localization mode.")
            return

        try:
            print(f"[INFO] Initializing EasyOCR Reader (languages: {self.languages})...")
            # verbose=False suppresses urllib progress output spam
            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
            print("[INFO] EasyOCR Reader initialized successfully.")
        except Exception as e:
            print(f"[WARNING] EasyOCR initialization deferred: {e}. Running ANPR with adaptive fallback.")

    def clean_plate_text(self, raw_text: str) -> str:
        """
        Perform normalization on raw OCR output text:
        - Convert to uppercase
        - Remove non-alphanumeric characters
        - Filter out noisy short fragments (< 3 characters)

        :param raw_text: Raw string output from OCR model.
        :return: Cleaned alphanumeric plate string.
        """
        if not raw_text:
            return ""
        cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        return cleaned if len(cleaned) >= 3 else ""

    def extract_plate_region(self, frame: np.ndarray, vehicle_bbox: List[int]) -> Tuple[np.ndarray, List[int]]:
        """
        Crop vehicle Region of Interest (ROI) and localize lower bumper license plate region.

        :param frame: Full image frame (BGR).
        :param vehicle_bbox: Vehicle bounding box [x1, y1, x2, y2].
        :return: Tuple of (cropped_plate_image, plate_bbox_relative_to_vehicle).
        """
        x1, y1, x2, y2 = vehicle_bbox
        vh, vw = max(1, y2 - y1), max(1, x2 - x1)

        # Vehicle ROI crop
        vehicle_roi = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]

        if vehicle_roi.size == 0:
            return np.array([]), [0, 0, 0, 0]

        # Plate candidate heuristic: license plates are located in lower 60% of vehicle box
        lower_roi = vehicle_roi[int(vh * 0.4):, :]

        # Grayscale & Canny edge enhancement for plate contour detection
        gray = cv2.cvtColor(lower_roi if lower_roi.size > 0 else vehicle_roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 200)

        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        best_plate_crop = lower_roi if lower_roi.size > 0 else vehicle_roi
        plate_rect = [0, int(vh * 0.4), vw, vh]

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = float(w) / max(1, float(h))

            # License plates typically have aspect ratio between 1.8 and 6.5
            if 1.8 <= aspect_ratio <= 6.5 and w > 25 and h > 10:
                plate_crop = lower_roi[y:y + h, x:x + w]
                if plate_crop.size > 0:
                    best_plate_crop = plate_crop
                    plate_rect = [x, int(vh * 0.4) + y, x + w, int(vh * 0.4) + y + h]
                    break

        return best_plate_crop, plate_rect

    def read_license_plate(self, frame: np.ndarray, vehicle_bbox: List[int]) -> Dict[str, Any]:
        """
        Extract plate region and run OCR to obtain license plate string and confidence.

        :param frame: Full image frame (BGR).
        :param vehicle_bbox: [x1, y1, x2, y2] bounding box.
        :return: Dict containing plate_number, raw_ocr_text, normalized_plate_text, ocr_confidence, confidence_category, plate_bbox.
        """
        default_result = {
            "plate_number": "",
            "raw_ocr_text": "",
            "normalized_plate_text": "",
            "ocr_confidence": 0.0,
            "confidence_category": "LOW CONFIDENCE",
            "plate_bbox": [],
        }

        try:
            plate_crop, plate_bbox = self.extract_plate_region(frame, vehicle_bbox)
            if plate_crop.size == 0:
                return default_result

            if self.reader is not None:
                results = self.reader.readtext(plate_crop)
                best_raw = ""
                best_cleaned = ""
                best_conf = 0.0

                for bbox, text, prob in results:
                    cleaned = self.clean_plate_text(text)
                    if cleaned and prob > best_conf:
                        best_raw = text
                        best_cleaned = cleaned
                        best_conf = float(prob)

                if best_cleaned and best_conf >= self.confidence_threshold:
                    conf_val = round(best_conf, 3)
                    cat = classify_ocr_confidence(conf_val)
                    return {
                        "plate_number": best_cleaned,
                        "raw_ocr_text": best_raw,
                        "normalized_plate_text": best_cleaned,
                        "ocr_confidence": conf_val,
                        "confidence_category": cat,
                        "plate_bbox": plate_bbox,
                    }

            return default_result

        except Exception as e:
            # Handle unreadable plate crops gracefully without crashing
            return default_result


# ================================================================================
# PHASE 1A: CONFIDENCE-AWARE ANPR & IDENTITY FUSION UTILITIES
# ================================================================================

OCR_CONFUSION_MAP = {
    "O": "0", "0": "0",
    "I": "1", "1": "1",
    "B": "8", "8": "8",
    "S": "5", "5": "5",
    "Z": "2", "2": "2",
}


def canonicalize_plate(plate: str) -> str:
    """Map ambiguous OCR characters to unified canonical representations."""
    cleaned = re.sub(r"[^A-Z0-9]", "", plate.upper())
    return "".join(OCR_CONFUSION_MAP.get(c, c) for c in cleaned)


def fuzzy_plate_similarity(plate1: str, plate2: str) -> float:
    """
    Compute fuzzy similarity score between two license plates considering common OCR confusion errors:
    O <-> 0, I <-> 1, B <-> 8, S <-> 5, Z <-> 2.
    """
    p1 = re.sub(r"[^A-Z0-9]", "", plate1.upper())
    p2 = re.sub(r"[^A-Z0-9]", "", plate2.upper())

    if not p1 or not p2:
        return 0.0
    if p1 == p2:
        return 1.0

    c1 = canonicalize_plate(p1)
    c2 = canonicalize_plate(p2)

    # Exact match on canonicalized plates (differs only by OCR confusion characters)
    if c1 == c2:
        return 0.98

    # Weighted Levenshtein edit distance
    len_max = max(len(p1), len(p2))
    dp = [[0.0] * (len(p2) + 1) for _ in range(len(p1) + 1)]
    for i in range(len(p1) + 1):
        dp[i][0] = float(i)
    for j in range(len(p2) + 1):
        dp[0][j] = float(j)

    for i in range(1, len(p1) + 1):
        for j in range(1, len(p2) + 1):
            ch1, ch2 = p1[i - 1], p2[j - 1]
            if ch1 == ch2:
                cost = 0.0
            elif OCR_CONFUSION_MAP.get(ch1) == OCR_CONFUSION_MAP.get(ch2):
                cost = 0.1
            else:
                cost = 1.0

            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost,
            )

    edit_dist = dp[len(p1)][len(p2)]
    similarity = max(0.0, 1.0 - (edit_dist / float(len_max)))
    return round(similarity, 3)


def classify_ocr_confidence(confidence: float) -> str:
    """Classify OCR confidence into HIGH, MEDIUM, or LOW confidence categories."""
    if confidence >= 0.80:
        return "HIGH CONFIDENCE"
    elif confidence >= 0.50:
        return "MEDIUM CONFIDENCE"
    else:
        return "LOW CONFIDENCE"


class VehicleIdentityFusion:
    """
    Confidence-Aware Cross-Camera Vehicle Identity Fusion Engine.
    Fuses license plate observations using fuzzy plate matching, spatial-temporal
    transition validation, and OCR confidence metrics into unified identity records.
    """

    def __init__(self, min_transition_time: float = 2.0, max_transition_time: float = 1800.0):
        self.min_transition_time = min_transition_time
        self.max_transition_time = max_transition_time

    def validate_spatial_temporal(
        self,
        camera_1: str,
        timestamp_1: str,
        camera_2: str,
        timestamp_2: str,
    ) -> Dict[str, Any]:
        """
        Validate cross-camera transition plausibility based on timestamps and camera IDs.
        """
        try:
            t1 = datetime.fromisoformat(timestamp_1)
            t2 = datetime.fromisoformat(timestamp_2)
            dt = (t2 - t1).total_seconds()
        except Exception:
            return {
                "valid": False,
                "time_diff_seconds": 0.0,
                "temporal_score": 0.0,
                "reason": "Invalid timestamp format",
            }

        # Case 1: Same camera node
        if camera_1 == camera_2:
            if abs(dt) < 5.0:
                return {"valid": True, "time_diff_seconds": abs(dt), "temporal_score": 1.0, "reason": "Same camera observation"}
            return {"valid": True, "time_diff_seconds": abs(dt), "temporal_score": 0.9, "reason": "Same camera repeat sighting"}

        # Case 2: Cross-camera transition
        if dt < 0:
            return {"valid": False, "time_diff_seconds": dt, "temporal_score": 0.0, "reason": "Negative time delta (reverse sequence)"}

        if dt < self.min_transition_time:
            return {"valid": False, "time_diff_seconds": dt, "temporal_score": 0.0, "reason": "Physically implausible speed (teleportation)"}

        if dt > self.max_transition_time:
            return {"valid": False, "time_diff_seconds": dt, "temporal_score": 0.2, "reason": "Time gap exceeds maximum plausible window"}

        # Valid transition within optimal window
        if dt <= 600.0:
            temporal_score = 1.0
        else:
            temporal_score = max(0.5, 1.0 - ((dt - 600.0) / (self.max_transition_time - 600.0)) * 0.5)

        return {
            "valid": True,
            "time_diff_seconds": round(dt, 2),
            "temporal_score": round(temporal_score, 3),
            "reason": "Physically plausible transition",
        }

    def fuse_identities(
        self,
        obs1: Dict[str, Any],
        obs2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Fuse two vehicle observations across cameras and generate identity confidence + evidence chain.

        EXACT FORMULA:
        Fused Identity Confidence = 0.35 * min(ocr_1, ocr_2) + 0.40 * similarity_score + 0.25 * temporal_score
        (If temporal transition is physically implausible, fused confidence is set to 0.0 to reject match)
        """
        plate1 = obs1.get("normalized_plate_text") or obs1.get("plate_number", "")
        plate2 = obs2.get("normalized_plate_text") or obs2.get("plate_number", "")

        ocr1 = float(obs1.get("ocr_confidence", 0.0))
        ocr2 = float(obs2.get("ocr_confidence", 0.0))

        cam1 = obs1.get("camera_id", "cam_01")
        cam2 = obs2.get("camera_id", "cam_02")

        t1 = obs1.get("timestamp", "")
        t2 = obs2.get("timestamp", "")

        # 1. Fuzzy similarity score
        similarity_score = fuzzy_plate_similarity(plate1, plate2)

        # 2. Spatial-Temporal validation
        st_val = self.validate_spatial_temporal(cam1, t1, cam2, t2)
        temporal_score = st_val["temporal_score"]

        # 3. Fused identity confidence score calculation
        min_ocr = min(ocr1, ocr2)

        if not st_val["valid"]:
            # Reject match if transition is physically impossible
            fused_confidence = 0.0
            is_match = False
        else:
            fused_confidence = round(0.35 * min_ocr + 0.40 * similarity_score + 0.25 * temporal_score, 3)
            # Threshold for valid cross-camera identity match
            is_match = fused_confidence >= 0.50 and similarity_score >= 0.65

        # Primary identity plate selection: choose highest OCR confidence plate or canonical plate
        if ocr1 >= ocr2:
            primary_plate = plate1 if plate1 != "UNREADABLE" else plate2
        else:
            primary_plate = plate2 if plate2 != "UNREADABLE" else plate1

        confidence_cat = classify_ocr_confidence(fused_confidence)

        evidence_chain = {
            "fused_plate_number": primary_plate,
            "fused_identity_confidence": fused_confidence,
            "confidence_category": confidence_cat,
            "is_valid_match": is_match,
            "evidence": {
                "source_camera": cam1,
                "destination_camera": cam2,
                "timestamps": {
                    "source": t1,
                    "destination": t2,
                    "time_diff_seconds": st_val["time_diff_seconds"],
                },
                "original_ocr_values": {
                    "source": obs1.get("raw_ocr_text", plate1),
                    "destination": obs2.get("raw_ocr_text", plate2),
                },
                "normalized_values": {
                    "source": plate1,
                    "destination": plate2,
                },
                "ocr_confidence": {
                    "source": ocr1,
                    "destination": ocr2,
                },
                "similarity_score": similarity_score,
                "temporal_score": temporal_score,
                "transition_reason": st_val["reason"],
                "evidence_summary": (
                    "plate similarity + temporal consistency + valid camera transition"
                    if is_match
                    else f"rejected match: {st_val['reason']}"
                ),
            },
        }
        return evidence_chain


class ANPRPipeline:
    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.4,
    ):
        """
        Integrated ANPR Pipeline associating vehicle detection, ByteTrack tracking,
        and EasyOCR license plate recognition.
        """
        self.anpr_engine = ANPREngine()
        # Plate recognition memory cache: tracking_id -> {"plate_number": str, "raw_ocr_text": str, "normalized_plate_text": str, "ocr_confidence": float, "confidence_category": str}
        self.plate_cache: Dict[int, Dict[str, Any]] = {}
        # Structured record log
        self.anpr_records: List[Dict[str, Any]] = []

    def process_tracked_vehicles(
        self,
        frame: np.ndarray,
        camera_id: str,
        tracked_objects: List[Dict[str, Any]],
        timestamp: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform license plate recognition for each tracked vehicle and associate results with tracking_id & camera_id.

        :param frame: Current frame array.
        :param camera_id: Identifier of camera stream.
        :param tracked_objects: List of objects tracked by ByteTrack.
        :param timestamp: ISO timestamp.
        :return: Updated tracked_objects list containing associated plate_number and ocr_confidence.
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()

        updated_objects = []

        for obj in tracked_objects:
            track_id = obj.get("track_id")
            bbox = obj.get("bbox")
            vehicle_class = obj.get("class_name")
            det_conf = obj.get("confidence")

            plate_number = ""
            raw_ocr = ""
            norm_ocr = ""
            ocr_conf = 0.0
            conf_cat = "LOW CONFIDENCE"

            # Check if plate has already been recognized for this persistent tracking_id
            if track_id is not None and track_id in self.plate_cache:
                cache_entry = self.plate_cache[track_id]
                plate_number = cache_entry["plate_number"]
                raw_ocr = cache_entry.get("raw_ocr_text", plate_number)
                norm_ocr = cache_entry.get("normalized_plate_text", plate_number)
                ocr_conf = cache_entry["ocr_confidence"]
                conf_cat = cache_entry.get("confidence_category", classify_ocr_confidence(ocr_conf))
            else:
                # Run ANPR OCR engine on vehicle crop
                ocr_res = self.anpr_engine.read_license_plate(frame, bbox)
                if ocr_res["plate_number"]:
                    plate_number = ocr_res["plate_number"]
                    raw_ocr = ocr_res.get("raw_ocr_text", plate_number)
                    norm_ocr = ocr_res.get("normalized_plate_text", plate_number)
                    ocr_conf = ocr_res["ocr_confidence"]
                    conf_cat = ocr_res.get("confidence_category", classify_ocr_confidence(ocr_conf))

                    if track_id is not None:
                        self.plate_cache[track_id] = {
                            "plate_number": plate_number,
                            "raw_ocr_text": raw_ocr,
                            "normalized_plate_text": norm_ocr,
                            "ocr_confidence": ocr_conf,
                            "confidence_category": conf_cat,
                        }

            # Create structured ANPR event record (retains backwards compatibility + new Phase 1A fields)
            record = {
                "camera_id": camera_id,
                "tracking_id": track_id,
                "plate_number": plate_number if plate_number else "UNREADABLE",
                "raw_ocr_text": raw_ocr if raw_ocr else ("UNREADABLE" if not plate_number else plate_number),
                "normalized_plate_text": norm_ocr if norm_ocr else ("UNREADABLE" if not plate_number else plate_number),
                "ocr_confidence": ocr_conf,
                "confidence_category": conf_cat if plate_number else "LOW CONFIDENCE",
                "vehicle_class": vehicle_class,
                "confidence": det_conf,
                "timestamp": timestamp,
                "bbox": bbox,
            }
            self.anpr_records.append(record)

            # Copy object dict and attach plate details
            obj_copy = dict(obj)
            obj_copy["plate_number"] = plate_number
            obj_copy["raw_ocr_text"] = raw_ocr
            obj_copy["normalized_plate_text"] = norm_ocr
            obj_copy["ocr_confidence"] = ocr_conf
            obj_copy["confidence_category"] = conf_cat if plate_number else "LOW CONFIDENCE"
            updated_objects.append(obj_copy)

        return updated_objects

    def draw_anpr_overlays(self, frame: np.ndarray, updated_objects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Render vehicle bounding box, tracking ID, vehicle class, and license plate number overlay.
        """
        output_frame = frame.copy()

        colors = {
            "car": (0, 255, 0),
            "motorcycle": (255, 165, 0),
            "bus": (0, 165, 255),
            "truck": (0, 0, 255),
        }

        for obj in updated_objects:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["class_name"].capitalize()
            track_id = obj.get("track_id")
            plate = obj.get("plate_number")
            color = colors.get(obj["class_name"], (255, 255, 255))

            # Bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            # Display tag: e.g. "ID:3 Car [MH12AB1234]"
            if plate:
                display_text = f"ID:{track_id} {label} [{plate}]"
            elif track_id is not None:
                display_text = f"ID:{track_id} {label}"
            else:
                display_text = f"{label}"

            (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            cv2.rectangle(
                output_frame,
                (x1, max(0, y1 - text_h - 8)),
                (x1 + text_w + 6, max(text_h + 8, y1)),
                color,
                -1,
            )

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


def main():
    parser = argparse.ArgumentParser(
        description="SIH26127 - ANPR & License Plate Recognition Engine."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="backend/videos/sample_traffic.mp4",
        help="Input source: MP4 video file path (default: 'backend/videos/sample_traffic.mp4') or webcam index '0'.",
    )
    parser.add_argument(
        "--multi-cam",
        action="store_true",
        help="Run ANPR in multi-camera test mode.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold. Default: 0.25.",
    )

    args = parser.parse_args()

    # Initialize ANPR pipeline and tracker
    pipeline = ANPRPipeline(confidence_threshold=args.conf)

class MultiCameraANPRTracker:
    def __init__(
        self,
        camera_sources: Dict[str, Union[str, int]],
        model_path: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        tracker_type: str = "bytetrack.yaml",
    ):
        """
        Multi-Camera ANPR Vehicle Tracker integrating YOLO detection, ByteTrack tracking,
        and EasyOCR license plate recognition across multiple simultaneous camera streams.

        :param camera_sources: Dict mapping camera_id to source (e.g. {'cam_01': 'backend/videos/sample.mp4', 'cam_02': 'backend/videos/sample_cam2.mp4'}).
        :param model_path: YOLO model weights path.
        :param confidence_threshold: Confidence threshold.
        :param tracker_type: Tracker configuration file.
        """
        self.camera_sources = camera_sources
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.tracker_type = tracker_type

        # Initialize ANPR Pipeline (shared OCR engine & cache)
        self.anpr_pipeline = ANPRPipeline(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
        )

        # Initialize independent SingleCameraTracker per camera feed
        self.trackers: Dict[str, SingleCameraTracker] = {}
        for camera_id in camera_sources:
            self.trackers[camera_id] = SingleCameraTracker(
                model_path=model_path,
                confidence_threshold=confidence_threshold,
                tracker_type=tracker_type,
            )

        # Global cross-camera vehicle trajectory mapping: plate_number -> List of sightings
        self.global_trajectories: Dict[str, List[Dict[str, Any]]] = {}

    def run_multi_anpr_stream(self) -> None:
        """
        Open all camera streams simultaneously, process vehicle detection, ByteTrack tracking,
        ANPR plate recognition, associate cross-camera vehicle sightings, and render feeds.
        """
        caps: Dict[str, cv2.VideoCapture] = {}

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
                print(f"[ERROR] MultiCameraANPRTracker: Failed to open camera '{camera_id}' source: '{source}'")
            else:
                caps[camera_id] = cap
                print(f"[INFO] Initialized ANPR Camera Node '{camera_id}' (Source: {source})")

        if not caps:
            print("[ERROR] No valid camera sources available.")
            return

        print(f"[INFO] Multi-Camera ANPR vehicle tracking active across {len(caps)} feeds.")
        print("[INFO] Press 'Q' or 'ESC' on any display window to exit.")

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

                    # Step 1: ByteTrack single-camera tracking
                    tracked_objects = tracker_instance.track_frame(frame)

                    # Step 2: ANPR license plate recognition
                    anpr_objects = self.anpr_pipeline.process_tracked_vehicles(
                        frame=frame,
                        camera_id=camera_id,
                        tracked_objects=tracked_objects,
                        timestamp=current_timestamp,
                    )

                    # Step 3: Cross-camera spatial-temporal plate association
                    for obj in anpr_objects:
                        plate = obj.get("plate_number")
                        if plate and plate != "UNREADABLE":
                            sighting = {
                                "camera_id": camera_id,
                                "local_track_id": obj.get("track_id"),
                                "vehicle_class": obj.get("class_name"),
                                "confidence": obj.get("confidence"),
                                "timestamp": current_timestamp,
                                "bbox": obj.get("bbox"),
                            }
                            if plate not in self.global_trajectories:
                                self.global_trajectories[plate] = []
                            self.global_trajectories[plate].append(sighting)

                    # Step 4: Render ANPR overlays
                    annotated_frame = self.anpr_pipeline.draw_anpr_overlays(frame, anpr_objects)

                    # Header banner with camera ID
                    cv2.putText(
                        annotated_frame,
                        f"CAM: {camera_id} | Tracked: {len(anpr_objects)}",
                        (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    cv2.imshow(f"SIH26127 Multi-Cam ANPR - {camera_id}", annotated_frame)

                if active_streams == 0:
                    print("[INFO] All video feeds finished.")
                    break

                if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                    print("[INFO] User requested exit.")
                    break

        except KeyboardInterrupt:
            print("[INFO] Multi-Camera ANPR stream interrupted by user.")
        finally:
            for cap in caps.values():
                cap.release()
            cv2.destroyAllWindows()
            print(f"[INFO] Multi-Camera ANPR stream closed.")
            print(f"[INFO] Total Unique License Plates Identified Across Cameras: {len(self.global_trajectories)}")
            for plate, sightings in self.global_trajectories.items():
                cams = list(set(s["camera_id"] for s in sightings))
                print(f"   -> Plate [{plate}]: Sightings={len(sightings)}, Cameras={cams}")


def main():
    parser = argparse.ArgumentParser(
        description="SIH26127 - Multi-Camera ANPR Vehicle Tracking Engine."
    )
    parser.add_argument(
        "--source",
        type=str,
        default="backend/videos/sample_traffic.mp4",
        help="Input source for single-cam: MP4 video path (default: 'backend/videos/sample_traffic.mp4') or webcam '0'.",
    )
    parser.add_argument(
        "--multi-cam",
        action="store_true",
        help="Run ANPR in multi-camera test mode across cam_01 and cam_02.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold. Default: 0.25.",
    )

    args = parser.parse_args()

    if args.multi_cam:
        print("[INFO] Launching Multi-Camera ANPR Vehicle Tracking Engine...")
        multi_anpr = MultiCameraANPRTracker(
            camera_sources={
                "cam_01": "backend/videos/sample_traffic.mp4",
                "cam_02": "backend/videos/sample_traffic_cam2.mp4",
            },
            confidence_threshold=args.conf,
        )
        multi_anpr.run_multi_anpr_stream()
    else:
        print(f"[INFO] Running Single-Camera ANPR Engine on source: '{args.source}'...")
        pipeline = ANPRPipeline(confidence_threshold=args.conf)
        single_tracker = SingleCameraTracker(confidence_threshold=args.conf)

        src = int(args.source) if args.source.isdigit() else args.source
        if isinstance(src, int):
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
            if not cap.isOpened() or not cap.grab():
                cap.release()
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)

        if not cap.isOpened():
            print(f"[ERROR] Could not open source: '{args.source}'")
            return

        print("[INFO] ANPR Stream active. Press 'Q' or 'ESC' on display window to quit.")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                tracked_objects = single_tracker.track_frame(frame)
                anpr_objects = pipeline.process_tracked_vehicles(frame, "cam_01", tracked_objects)
                annotated_frame = pipeline.draw_anpr_overlays(frame, anpr_objects)

                cv2.imshow("SIH26127 - ANPR License Plate Recognition", annotated_frame)
                if (cv2.waitKey(1) & 0xFF) in (ord('q'), ord('Q'), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print(f"[INFO] ANPR Stream ended. Total ANPR records logged: {len(pipeline.anpr_records)}")


if __name__ == "__main__":
    main()
