#!/usr/bin/env python3

from ultralytics import YOLO
import cv2
import time
from collections import deque
from datetime import datetime
from auppbot import AUPPBot  # Import AUPP robot library

# ============================================
# CONFIGURATION - ADJUST FOR YOUR SETUP
# ============================================

CONFIG = {
    # Model Settings
    'MODEL_PATH': 'runs/detect/train2/weights/best.pt',
    'CONFIDENCE': 0.40,
    'MIN_CONF_ACTION': 0.45,
    'STABLE_FRAMES': 2,
    
    # Class to Action Mapping
    'CLASSES': {
        0: 'Peach_Pit',
        1: 'Toilet_Paper', 
        2: 'Cigarette_Butts',
        3: 'Disposable_Chopstick'
    },
    
    'ACTIONS': {
        0: 'MOVE FORWARD',
        1: 'MOVE BACKWARD',
        2: 'TURN LEFT',
        3: 'TURN RIGHT'
    },
    
    # AUPP Robot Serial Connection
    'SERIAL_PORT': '/dev/ttyUSB0',  # Change if needed: /dev/ttyACM0, /dev/ttyUSB0, etc.
    'BAUD_RATE': 115200,
    
    # Movement Parameters (AUPP uses 0-99 speed scale)
    'BASE_SPEED': 30,        # Base speed for forward movement
    'TURN_SPEED': 25,        # Speed when turning
    'TURN_DURATION': 0.5,    # How long to turn (seconds)
    'MOVE_DURATION': 0.7,    # How long to move forward/backward
    'SEARCH_SPEED': 20,      # Speed when searching
    'SEARCH_DURATION': 0.3,  # Search movement duration
    
    # Detection Settings
    'NO_DETECT_FRAMES': 15,  # Frames before searching
    'ACTION_COOLDOWN': 0.4,  # Min seconds between actions
    
    # Camera
    'CAMERA_INDEX': 0,       # Usually 0 for Raspberry Pi camera
    'CAMERA_WIDTH': 640,
    'CAMERA_HEIGHT': 480,
}

# ============================================
# AUPP ROBOT CONTROLLER
# ============================================

class AUPPRobotController:
    """Controls AUPP robot using auppbot library"""
    
    def __init__(self):
        self.last_action = None
        self.last_action_time = 0
        self.total_actions = 0
        self.search_moves = 0
        self.action_log = deque(maxlen=50)
        self.bot = None
        
        # Initialize AUPP robot
        try:
            print(f"🔌 Connecting to AUPP robot on {CONFIG['SERIAL_PORT']}...")
            self.bot = AUPPBot(
                port=CONFIG['SERIAL_PORT'],
                baud=CONFIG['BAUD_RATE'],
                auto_safe=True,
                timeout=1
            )
            print("✅ AUPP robot connected!")
            
            # Center servos at startup
            try:
                self.bot.servo1.angle(90)
                self.bot.servo2.angle(90)
                print("✅ Servos centered")
            except:
                print("⚠️  Could not center servos")
            
        except Exception as e:
            print(f"❌ Failed to connect to AUPP robot: {e}")
            print("   Make sure:")
            print(f"   1. Robot is connected to {CONFIG['SERIAL_PORT']}")
            print("   2. You have permission: sudo usermod -a -G dialout $USER")
            print("   3. No other program is using the serial port")
            self.bot = None
    
    def stop_motors(self):
        """Stop all motors"""
        if self.bot:
            self.bot.motor1.stop()
            self.bot.motor2.stop()
            self.bot.motor3.stop()
            self.bot.motor4.stop()
    
    def move_forward(self, duration):
        """Move robot forward - both sides same speed"""
        if not self.bot:
            print("   ⚠️  (Simulated - Robot not connected)")
            return
        
        speed = CONFIG['BASE_SPEED']
        # Motors 1&2 = left side, Motors 3&4 = right side
        self.bot.motor1.forward(speed)
        self.bot.motor2.forward(speed)
        self.bot.motor3.forward(speed)
        self.bot.motor4.forward(speed)
        time.sleep(duration)
        self.stop_motors()
    
    def move_backward(self, duration):
        """Move robot backward - both sides same speed"""
        if not self.bot:
            print("   ⚠️  (Simulated - Robot not connected)")
            return
        
        speed = CONFIG['BASE_SPEED']
        self.bot.motor1.backward(speed)
        self.bot.motor2.backward(speed)
        self.bot.motor3.backward(speed)
        self.bot.motor4.backward(speed)
        time.sleep(duration)
        self.stop_motors()
    
    def turn_left(self, duration):
        """Turn robot left - left side backward, right side forward"""
        if not self.bot:
            print("   ⚠️  (Simulated - Robot not connected)")
            return
        
        speed = CONFIG['TURN_SPEED']
        # Left side backward, right side forward
        self.bot.motor1.backward(speed)
        self.bot.motor2.backward(speed)
        self.bot.motor3.forward(speed)
        self.bot.motor4.forward(speed)
        time.sleep(duration)
        self.stop_motors()
    
    def turn_right(self, duration):
        """Turn robot right - left side forward, right side backward"""
        if not self.bot:
            print("   ⚠️  (Simulated - Robot not connected)")
            return
        
        speed = CONFIG['TURN_SPEED']
        # Left side forward, right side backward
        self.bot.motor1.forward(speed)
        self.bot.motor2.forward(speed)
        self.bot.motor3.backward(speed)
        self.bot.motor4.backward(speed)
        time.sleep(duration)
        self.stop_motors()
    
    def can_execute(self, action_type):
        """Check if enough time passed since last action"""
        now = time.time()
        if self.last_action == action_type:
            if (now - self.last_action_time) < CONFIG['ACTION_COOLDOWN']:
                return False
        return True
    
    def execute_detection_action(self, class_id, class_name, confidence):
        """Execute action based on detected class"""
        
        action_name = CONFIG['ACTIONS'][class_id]
        
        if not self.can_execute(f"detect_{class_id}"):
            return
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Print detection info
        print("\n" + "="*70)
        print(f"🎯 [{timestamp}] DETECTED: {class_name} (conf: {confidence:.2f})")
        print(f"📋 ACTION: {action_name}")
        
        # Execute the corresponding movement
        if class_id == 0:  # Peach_Pit → TURN LEFT
            print(f"   ↻ Turning LEFT for {CONFIG['TURN_DURATION']}s")
            self.turn_left(CONFIG['TURN_DURATION'])
        
        elif class_id == 1:  # Toilet_Paper → TURN RIGHT
            print(f"   ↺ Turning RIGHT for {CONFIG['TURN_DURATION']}s")
            self.turn_right(CONFIG['TURN_DURATION'])
        
        elif class_id == 2:  # Cigarette_Butts → MOVE FORWARD
            print(f"   ↑ Moving FORWARD for {CONFIG['MOVE_DURATION']}s")
            self.move_forward(CONFIG['MOVE_DURATION'])
        
        elif class_id == 3:  # Disposable_Chopstick → MOVE BACKWARD
            print(f"   ↓ Moving BACKWARD for {CONFIG['MOVE_DURATION']}s")
            self.move_backward(CONFIG['MOVE_DURATION'])
        
        print("✅ Action completed!")
        print("="*70)
        
        # Update tracking
        self.last_action = f"detect_{class_id}"
        self.last_action_time = time.time()
        self.total_actions += 1
        
        self.action_log.append({
            'time': timestamp,
            'type': 'detection',
            'class_id': class_id,
            'class_name': class_name,
            'action': action_name,
            'confidence': confidence
        })
    
    def search_forward(self):
        """Move forward when no detection (searching)"""
        if not self.can_execute("search"):
            return
        
        print(f"🔍 No detection - Moving forward to search...")
        
        if self.bot:
            speed = CONFIG['SEARCH_SPEED']
            self.bot.motor1.forward(speed)
            self.bot.motor2.forward(speed)
            self.bot.motor3.forward(speed)
            self.bot.motor4.forward(speed)
            time.sleep(CONFIG['SEARCH_DURATION'])
            self.stop_motors()
        
        self.last_action = "search"
        self.last_action_time = time.time()
        self.search_moves += 1
    
    def cleanup(self):
        """Cleanup robot on exit"""
        if self.bot:
            print("🧹 Stopping robot and cleaning up...")
            self.stop_motors()
            try:
                self.bot.safe()
                self.bot.close()
            except:
                pass
            print("✅ Robot stopped safely")
    
    def get_stats(self):
        """Get action statistics"""
        stats = {}
        for entry in self.action_log:
            if entry['type'] == 'detection':
                action = entry['action']
                stats[action] = stats.get(action, 0) + 1
        return stats

# ============================================
# DETECTION STABILIZER
# ============================================

class DetectionStabilizer:
    """Ensures consistent detection before taking action"""
    
    def __init__(self, required_frames=2):
        self.required_frames = required_frames
        self.history = deque(maxlen=required_frames)
        self.fps_history = deque(maxlen=30)
    
    def add_detection(self, class_id):
        """Add a detection to history"""
        self.history.append(class_id)
    
    def is_stable(self):
        """Check if we have consistent detections"""
        if len(self.history) < self.required_frames:
            return False, None
        
        # Check if all recent detections are the same class
        if len(set(self.history)) == 1:
            return True, self.history[-1]
        
        return False, None
    
    def clear(self):
        """Clear detection history"""
        self.history.clear()
    
    def update_fps(self, frame_time):
        """Track FPS"""
        self.fps_history.append(frame_time)
    
    def get_fps(self):
        """Calculate average FPS"""
        if not self.fps_history:
            return 0
        return 1.0 / (sum(self.fps_history) / len(self.fps_history))

# ============================================
# MAIN PROGRAM
# ============================================

def main():
    """Main robot control loop"""
    
    print("\n" + "="*70)
    print("🤖 AUPP CUBE FOLLOWER ROBOT - YOLO Detection")
    print("="*70)
    
    print("\n📦 BEHAVIOR:")
    print("   🔍 NO DETECTION     → Robot moves FORWARD (searching)")
    print("   🎯 DETECTION FOUND  → Robot performs action:")
    
    for class_id in sorted(CONFIG['CLASSES'].keys()):
        class_name = CONFIG['CLASSES'][class_id]
        action = CONFIG['ACTIONS'][class_id]
        print(f"      [{class_id}] {class_name:22s} → {action}")
    
    print("\n⚙️  MOVEMENT SETTINGS:")
    print(f"   Base Speed: {CONFIG['BASE_SPEED']}/99")
    print(f"   Turn Speed: {CONFIG['TURN_SPEED']}/99")
    print(f"   Turn Duration: {CONFIG['TURN_DURATION']}s")
    print(f"   Move Duration: {CONFIG['MOVE_DURATION']}s")
    
    print("\n🔌 SERIAL CONNECTION:")
    print(f"   Port: {CONFIG['SERIAL_PORT']}")
    print(f"   Baud: {CONFIG['BAUD_RATE']}")
    
    print("="*70)
    
    # Check model exists
    import os
    if not os.path.exists(CONFIG['MODEL_PATH']):
        print(f"\n❌ ERROR: Model not found at {CONFIG['MODEL_PATH']}")
        return
    
    try:
        # Load YOLO model
        print(f"\n📥 Loading YOLO model: {CONFIG['MODEL_PATH']}")
        model = YOLO(CONFIG['MODEL_PATH'])
        model.conf = CONFIG['CONFIDENCE']
        print("✅ Model loaded successfully")
        
        # Initialize robot
        robot = AUPPRobotController()
        if not robot.bot:
            print("\n❌ Cannot continue without robot connection")
            return
        
        stabilizer = DetectionStabilizer(CONFIG['STABLE_FRAMES'])
        
        # Open camera
        print(f"\n📹 Opening camera {CONFIG['CAMERA_INDEX']}...")
        cap = cv2.VideoCapture(CONFIG['CAMERA_INDEX'])
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CONFIG['CAMERA_WIDTH'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG['CAMERA_HEIGHT'])
        print("✅ Camera ready")
        
        print("\n🚀 ROBOT STARTING!")
        print("   - Robot will move FORWARD when no cube detected")
        print("   - Robot will perform actions when cube detected")
        print("   - Press Ctrl+C to stop")
        print("="*70 + "\n")
        
        frame_count = 0
        no_detect_count = 0
        last_status_time = time.time()
        
        while True:
            loop_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
            
            frame_count += 1
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            detections = results[0].boxes
            
            detected = False
            
            if len(detections) > 0:
                # Get best detection
                best_idx = detections.conf.argmax()
                conf = float(detections.conf[best_idx])
                class_id = int(detections.cls[best_idx])
                
                # Check if confidence is high enough
                if conf >= CONFIG['MIN_CONF_ACTION']:
                    # Add to stabilizer
                    stabilizer.add_detection(class_id)
                    
                    # Check if detection is stable
                    is_stable, stable_class = stabilizer.is_stable()
                    
                    if is_stable:
                        # Execute action!
                        class_name = CONFIG['CLASSES'][stable_class]
                        robot.execute_detection_action(stable_class, class_name, conf)
                        stabilizer.clear()
                        no_detect_count = 0
                        detected = True
                    else:
                        # Still stabilizing
                        class_name = CONFIG['CLASSES'][class_id]
                        print(f"⏳ Stabilizing {class_name}... ({len(stabilizer.history)}/{CONFIG['STABLE_FRAMES']})", end='\r')
                        detected = True
            
            if not detected:
                # No detection - increment counter
                stabilizer.clear()
                no_detect_count += 1
                
                # Move forward to search
                if no_detect_count >= CONFIG['NO_DETECT_FRAMES']:
                    robot.search_forward()
                    no_detect_count = 0
            
            # Print status every 10 seconds
            if time.time() - last_status_time > 10:
                fps = stabilizer.get_fps()
                print(f"\n📊 Status: Frame {frame_count} | FPS: {fps:.1f} | Actions: {robot.total_actions} | Searches: {robot.search_moves}")
                last_status_time = time.time()
            
            # Track FPS
            stabilizer.update_fps(time.time() - loop_start)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Stopped by user (Ctrl+C)")
    
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        try:
            cap.release()
        except:
            pass
        
        robot.cleanup()
        
        # Print final statistics
        print("\n" + "="*70)
        print("📊 SESSION SUMMARY")
        print("="*70)
        print(f"   Total Frames: {frame_count}")
        print(f"   Detection Actions: {robot.total_actions}")
        print(f"   Search Moves: {robot.search_moves}")
        print(f"   Average FPS: {stabilizer.get_fps():.1f}")
        
        stats = robot.get_stats()
        if stats:
            print("\n   Detection Action Breakdown:")
            for action, count in sorted(stats.items()):
                print(f"      {action}: {count}")
        
        if len(robot.action_log) > 0:
            print("\n   Recent Actions:")
            for entry in list(robot.action_log)[-5:]:
                if entry['type'] == 'detection':
                    print(f"      [{entry['time']}] {entry['class_name']} → {entry['action']}")
        
        print("="*70)
        print("\n✅ Robot stopped successfully!\n")

if __name__ == "__main__":
    print("\n💡 AUPP ROBOT SETUP:")
    print("   - Make sure robot is connected via USB")
    print("   - Check serial port in config (default: /dev/ttyUSB0)")
    print("   - Motors 1&2 = Left side, Motors 3&4 = Right side")
    print("   - Press Ctrl+C to stop safely\n")
    
    main()
