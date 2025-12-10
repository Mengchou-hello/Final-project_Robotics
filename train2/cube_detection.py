#!/usr/bin/env python3
"""
Cube Detector - Live Camera Test
Shows real-time detection with bounding boxes
Press 'q' to quit
"""

from ultralytics import YOLO
import cv2

# ===========================
# Configuration
# ===========================
MODEL_PATH = "//Users/mengchou/Downloads/Hazardous-Waste/runs/detect/train2/cube_detection/weights/best.pt"
CONFIDENCE = 0.35
CAMERA_INDEX = 0

# Class names for display
CLASS_NAMES = {
    0: 'Peach_Pit',
    1: 'Toilet_Paper',
    2: 'Cigarette_Butts',
    3: 'Disposable_Chopstick'
}

print("="*60)
print("🎥 CUBE DETECTOR - LIVE CAMERA TEST")
print("="*60)

# ===========================
# Load Model
# ===========================
print(f"\n📥 Loading model: {MODEL_PATH}")
try:
    model = YOLO(MODEL_PATH)
    model.conf = CONFIDENCE
    print(f"✅ Model loaded (confidence: {CONFIDENCE})")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# ===========================
# Initialize Camera
# ===========================
print(f"\n📹 Opening camera {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("❌ Error: Cannot open camera")
    print("   Try: ls /dev/video*")
    exit(1)

# Get camera info
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✅ Camera ready: {width}x{height}")

print("\n🚀 STARTING DETECTION")
print("   - Show cube to camera")
print("   - Press 'q' to quit")
print("="*60 + "\n")

# ===========================
# Live Detection Loop
# ===========================
frame_count = 0

try:
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame")
            break
        
        frame_count += 1
        
        # Run YOLO detection
        results = model(frame, verbose=False)
        
        # Get annotated frame with bounding boxes
        annotated_frame = results[0].plot()
        
        # Display detection info on frame
        detections = results[0].boxes
        if len(detections) > 0:
            # Get best detection
            best_idx = detections.conf.argmax()
            conf = float(detections.conf[best_idx])
            class_id = int(detections.cls[best_idx])
            class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
            
            # Print to console
            if frame_count % 10 == 0:  # Every 10 frames
                print(f"🎯 Detected: {class_name} ({conf:.2%})")
        
        # Show frame
        cv2.imshow("Cube Detector - Press Q to Quit", annotated_frame)
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️  Quit key pressed")
            break

except KeyboardInterrupt:
    print("\n⚠️  Interrupted by user (Ctrl+C)")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # ===========================
    # Cleanup
    # ===========================
    print("\n🧹 Cleaning up...")
    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Processed {frame_count} frames")
    print("✅ Done!\n")