"""
License Plate Detection - Live Webcam Runner
============================================
Real-time license plate detection using YOLOv11 model with webcam feed.

Features:
- Live webcam detection with bounding boxes
- FPS counter
- Adjustable confidence threshold
- Keyboard controls for interaction
- Multiple camera support

Controls:
- Press 'q' or ESC: Quit
- Press '+': Increase confidence threshold
- Press '-': Decrease confidence threshold
- Press 's': Save current frame
- Press 'r': Reset confidence to default

Author: Plate Detector Project
"""

import cv2
import time
from ultralytics import YOLO
from pathlib import Path
import sys

class LicensePlateDetector:
    """Real-time license plate detector using YOLO model."""
    
    def __init__(self, model_path, conf_threshold=0.4, camera_id=0):
        """
        Initialize the detector.
        
        Args:
            model_path (str): Path to the YOLO model weights
            conf_threshold (float): Initial confidence threshold (0.0 - 1.0)
            camera_id (int): Camera device ID (0 for default webcam)
        """
        self.conf_threshold = conf_threshold
        self.default_conf = conf_threshold
        self.camera_id = camera_id
        self.frame_count = 0
        self.saved_count = 0
        
        # Initialize model
        print(f"Loading YOLO model from: {model_path}")
        try:
            self.model = YOLO(model_path)
            print("✓ Model loaded successfully!")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)
        
        # Create output directory for saved frames
        self.output_dir = Path("runs/webcam_detections")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def run(self):
        """Start the webcam detection loop."""
        # Open webcam
        print(f"\nOpening camera (ID: {self.camera_id})...")
        cap = cv2.VideoCapture(self.camera_id)
        
        if not cap.isOpened():
            print(f"✗ Error: Could not open camera {self.camera_id}")
            print("Tip: Try changing camera_id (0, 1, 2, etc.)")
            sys.exit(1)
        
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✓ Camera opened successfully!")
        print("\n" + "="*60)
        print("LIVE DETECTION STARTED")
        print("="*60)
        self._print_controls()
        
        # FPS calculation variables
        fps = 0
        frame_time = time.time()
        fps_update_interval = 0.5  # Update FPS every 0.5 seconds
        last_fps_update = time.time()
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    print("✗ Failed to grab frame")
                    break
                
                self.frame_count += 1
                
                # Run YOLO detection
                results = self.model(frame, conf=self.conf_threshold, verbose=False)
                
                # Get annotated frame with bounding boxes
                annotated_frame = results[0].plot()
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_fps_update >= fps_update_interval:
                    fps = 1 / (current_time - frame_time) if (current_time - frame_time) > 0 else 0
                    last_fps_update = current_time
                frame_time = current_time
                
                # Count detections
                num_detections = len(results[0].boxes)
                
                # Display information overlay
                self._draw_info_overlay(annotated_frame, fps, num_detections)
                
                # Show the frame
                cv2.imshow("License Plate Detection - Live", annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_key_press(key, annotated_frame):
                    break  # Exit if 'q' or ESC pressed
                    
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted by user (Ctrl+C)")
        finally:
            # Cleanup
            print("\n" + "="*60)
            print(f"Total frames processed: {self.frame_count}")
            print(f"Frames saved: {self.saved_count}")
            print("="*60)
            cap.release()
            cv2.destroyAllWindows()
            print("✓ Cleanup complete. Exiting...")
    
    def _draw_info_overlay(self, frame, fps, num_detections):
        """Draw information overlay on frame."""
        height, width = frame.shape[:2]
        
        # Semi-transparent background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text information
        info_text = [
            f"FPS: {fps:.1f}",
            f"Confidence: {self.conf_threshold:.2f}",
            f"Detections: {num_detections}",
            f"Frame: {self.frame_count}"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            color = (0, 255, 0) if i == 2 and num_detections > 0 else (255, 255, 255)
            cv2.putText(frame, text, (20, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Status message at bottom
        status = "PLATE DETECTED!" if num_detections > 0 else "Scanning..."
        status_color = (0, 255, 0) if num_detections > 0 else (100, 100, 100)
        cv2.putText(frame, status, (20, height - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    def _handle_key_press(self, key, frame):
        """
        Handle keyboard input.
        
        Returns:
            bool: True to continue, False to exit
        """
        # Quit: 'q' or ESC
        if key == ord('q') or key == 27:
            return False
        
        # Increase confidence: '+'
        elif key == ord('+') or key == ord('='):
            self.conf_threshold = min(1.0, self.conf_threshold + 0.05)
            print(f"\n→ Confidence increased to {self.conf_threshold:.2f}")
        
        # Decrease confidence: '-'
        elif key == ord('-') or key == ord('_'):
            self.conf_threshold = max(0.1, self.conf_threshold - 0.05)
            print(f"\n→ Confidence decreased to {self.conf_threshold:.2f}")
        
        # Save frame: 's'
        elif key == ord('s'):
            self.saved_count += 1
            filename = self.output_dir / f"detection_{self.saved_count:04d}.jpg"
            cv2.imwrite(str(filename), frame)
            print(f"\n📷 Frame saved: {filename}")
        
        # Reset confidence: 'r'
        elif key == ord('r'):
            self.conf_threshold = self.default_conf
            print(f"\n↺ Confidence reset to {self.conf_threshold:.2f}")
        
        return True
    
    def _print_controls(self):
        """Print keyboard controls to console."""
        print("\n KEYBOARD CONTROLS:")
        print("  Q or ESC  : Quit")
        print("  + or =    : Increase confidence threshold")
        print("  - or _    : Decrease confidence threshold")
        print("  S         : Save current frame")
        print("  R         : Reset confidence to default")
        print("\n" + "="*60 + "\n")


def main():
    """Main entry point."""
    # Configuration
    MODEL_PATH = "training/runs/plate_detector_yolo113/weights/best.pt"
    CONFIDENCE_THRESHOLD = 0.4
    CAMERA_ID = 0
    
    # Check if model exists
    if not Path(MODEL_PATH).exists():
        print(f"✗ Error: Model file not found at: {MODEL_PATH}")
        print("\nAvailable model paths:")
        print("  - training/runs/plate_detector_yolo113/weights/best.pt")
        print("  - training/runs/plate_detector_yolo113/weights/last.pt")
        print("\nPlease update MODEL_PATH in the script.")
        sys.exit(1)
    
    # Create and run detector
    detector = LicensePlateDetector(
        model_path=MODEL_PATH,
        conf_threshold=CONFIDENCE_THRESHOLD,
        camera_id=CAMERA_ID
    )
    
    detector.run()


if __name__ == "__main__":
    main()
