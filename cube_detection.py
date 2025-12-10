#!/usr/bin/env python3

from ultralytics import YOLO
import cv2
import time
from datetime import datetime
import os

CONFIG = {
    'MODEL_PATH': 'runs/detect/train2/weights/best.pt',
    'CONFIDENCE': 0.40,
    'CLASSES': {
        0: 'Peach_Pit',
        1: 'Toilet_Paper',
        2: 'Cigarette_Butts',
        3: 'Disposable_Chopstick'
    },
    'ACTIONS': {
        0: '⬅️  TURN LEFT',
        1: '➡️  TURN RIGHT',
        2: '⬆️  MOVE FORWARD',
        3: '⬇️  MOVE BACKWARD'
    },
    'CAMERA_INDEX': 0,
    'FRAME_WIDTH': 640,
    'FRAME_HEIGHT': 480,
    'SHOW_FPS': True
}

class DetectionViewer:
    def __init__(self):
        self.frame_count = 0
        self.detection_count = 0
        self.fps = 0
        self.last_time = time.time()

    def draw_info(self, frame):
        cv2.putText(frame, f"FPS: {self.fps:.1f}",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        return frame

    def update_fps(self):
        now = time.time()
        self.fps = 1 / (now - self.last_time) if (now - self.last_time) > 0 else 0
        self.last_time = now


def main():
    print("\n📦 Checking model path...")
    if not os.path.exists(CONFIG['MODEL_PATH']):
        print(f"❌ Model not found at: {CONFIG['MODEL_PATH']}")
        return

    print("📥 Loading YOLO model...")
    model = YOLO(CONFIG['MODEL_PATH'])
    model.conf = CONFIG['CONFIDENCE']
    print("✅ Model loaded")

    print("\n📹 Opening camera...")
    cap = cv2.VideoCapture(CONFIG['CAMERA_INDEX'])
    if not cap.isOpened():
        print("❌ Camera cannot be opened")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['FRAME_WIDTH'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['FRAME_HEIGHT'])
    print("✅ Camera ready")

    print("\n🚀 Starting detection...")
    viewer = DetectionViewer()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break

        viewer.frame_count += 1

        results = model(frame, verbose=False)

        # Count detections
        if len(results[0].boxes) > 0:
            viewer.detection_count += 1

        # *** THIS DRAWS ALL BOXES + LABELS + CONFIDENCE ***
        display = results[0].plot()

        # Draw FPS
        viewer.update_fps()
        display = viewer.draw_info(display)

        cv2.imshow("Cube Detection Viewer", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"screenshot_{ts}.jpg", display)
            print(f"📸 Screenshot saved: screenshot_{ts}.jpg")

    cap.release()
    cv2.destroyAllWindows()

    print("\n📊 SUMMARY")
    print(f"Frames processed: {viewer.frame_count}")
    print(f"Frames with detections: {viewer.detection_count}")
    print("Viewer closed.")
    

if __name__ == "__main__":
    main()
