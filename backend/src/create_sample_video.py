"""
Sample Traffic Video Generator for SIH26127 Prototype Testing
--------------------------------------------------------------
Generates sample MP4 video files with visually detailed synthetic vehicles (cars, trucks, buses)
complete with wheels, windshields, and headlights so YOLO object detection models trigger reliably.
"""

import os
import cv2
import numpy as np


def draw_detailed_car(frame, x, y, width, height, body_color, is_truck=False, is_bus=False, plate_text=""):
    # Body fill
    cv2.rectangle(frame, (x, y), (x + width, y + height), body_color, -1)
    cv2.rectangle(frame, (x, y), (x + width, y + height), (30, 30, 30), 2)

    # Windshield / Windows
    if is_bus:
        win_w = int(width * 0.18)
        for i in range(4):
            wx = x + 10 + i * (win_w + 5)
            cv2.rectangle(frame, (wx, y + 10), (wx + win_w, y + int(height * 0.4)), (200, 230, 255), -1)
    elif is_truck:
        cv2.rectangle(frame, (x + int(width * 0.7), y + 5), (x + width - 5, y + height - 5), (180, 210, 240), -1)
    else:
        roof_x1 = x + int(width * 0.2)
        roof_x2 = x + int(width * 0.7)
        cv2.rectangle(frame, (roof_x1, y + 5), (roof_x2, y + int(height * 0.5)), (180, 220, 255), -1)

    # Wheels (dark circles at bottom)
    wheel_radius = int(height * 0.2)
    wheel_y = y + height
    cv2.circle(frame, (x + int(width * 0.25), wheel_y), wheel_radius, (20, 20, 20), -1)
    cv2.circle(frame, (x + int(width * 0.75), wheel_y), wheel_radius, (20, 20, 20), -1)
    cv2.circle(frame, (x + int(width * 0.25), wheel_y), int(wheel_radius * 0.4), (200, 200, 200), -1)
    cv2.circle(frame, (x + int(width * 0.75), wheel_y), int(wheel_radius * 0.4), (200, 200, 200), -1)

    # Headlights & Taillights
    cv2.rectangle(frame, (x + width - 6, y + int(height * 0.6)), (x + width, y + int(height * 0.8)), (0, 255, 255), -1)
    cv2.rectangle(frame, (x, y + int(height * 0.6)), (x + 6, y + int(height * 0.8)), (0, 0, 255), -1)

    # License Plate Badge (White background rectangle with dark bold plate text)
    plate_x = x + int(width * 0.25)
    plate_y = y + int(height * 0.65)
    pw = max(70, len(plate_text) * 8 + 10)
    cv2.rectangle(frame, (plate_x, plate_y), (plate_x + pw, plate_y + 20), (255, 255, 255), -1)
    cv2.rectangle(frame, (plate_x, plate_y), (plate_x + pw, plate_y + 20), (0, 0, 0), 2)
    if plate_text:
        cv2.putText(frame, plate_text, (plate_x + 4, plate_y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)


def generate_sample_traffic_video(output_path: str = "backend/videos/sample_traffic.mp4", duration_sec: int = 10, fps: int = 30, camera_offset: int = 0):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 960, 540
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    num_frames = duration_sec * fps
    print(f"[INFO] Generating realistic sample traffic video '{output_path}' ({num_frames} frames)...")

    vehicles = [
        {"x": 50 + camera_offset, "y": 180, "vx": 4, "w": 140, "h": 65, "color": (0, 180, 0), "type": "car", "plate": "MH12DE1408"},
        {"x": 100 + camera_offset, "y": 300, "vx": 5, "w": 180, "h": 85, "color": (40, 100, 220), "type": "truck", "plate": "KA01AB2026"},
        {"x": 800 - camera_offset, "y": 410, "vx": -5, "w": 130, "h": 60, "color": (220, 120, 0), "type": "car", "plate": "DL03XY9988"},
        {"x": 700 - camera_offset, "y": 130, "vx": -3, "w": 200, "h": 90, "color": (0, 200, 200), "type": "bus", "plate": "TN09CC5544"},
    ]

    for frame_idx in range(num_frames):
        frame = np.ones((height, width, 3), dtype=np.uint8) * (50 + (camera_offset % 20))
        
        # Road markings
        cv2.line(frame, (0, 110), (width, 110), (255, 255, 255), 3)
        cv2.line(frame, (0, 260), (width, 260), (200, 200, 200), 2, cv2.LINE_AA)
        cv2.line(frame, (0, 490), (width, 490), (255, 255, 255), 3)

        cam_label = os.path.basename(output_path).replace(".mp4", "").upper()
        cv2.putText(frame, f"FEED: {cam_label}", (width - 320, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        for v in vehicles:
            v["x"] += v["vx"]
            if v["vx"] > 0 and v["x"] > width:
                v["x"] = -v["w"]
            elif v["vx"] < 0 and v["x"] < -v["w"]:
                v["x"] = width

            x, y, w, h = int(v["x"]), int(v["y"]), v["w"], v["h"]
            draw_detailed_car(
                frame, x, y, w, h, v["color"],
                is_truck=(v["type"] == "truck"),
                is_bus=(v["type"] == "bus"),
                plate_text=v.get("plate", ""),
            )

        out.write(frame)

    out.release()
    print(f"[INFO] Sample traffic video saved successfully to: '{output_path}'")


def main():
    generate_sample_traffic_video("backend/videos/sample_traffic.mp4", duration_sec=10, camera_offset=0)
    generate_sample_traffic_video("backend/videos/sample_traffic_cam2.mp4", duration_sec=10, camera_offset=120)


if __name__ == "__main__":
    main()
