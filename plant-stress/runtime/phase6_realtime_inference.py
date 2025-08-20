#!/usr/bin/env python3
"""
Phase 6: Real-time Inference System
iPhone camera → Python via Continuity Camera for plant stress detection
"""

import os
import sys
import time
import argparse
import json
import threading
from pathlib import Path
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

import ultralytics
from ultralytics import YOLO

import timm
from timm.data import create_transform

class ContinuityCameraManager:
    """Manager for iPhone Continuity Camera integration"""
    
    def __init__(self):
        self.available_devices = []
        self.current_device = None
        self.cap = None
        
    def list_devices(self):
        """List available video devices using FFmpeg"""
        print("🔍 Enumerating video devices...")
        
        try:
            # Try FFmpeg to list devices
            import subprocess
            result = subprocess.run([
                'ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                print("📱 Available video devices:")
                lines = result.stderr.split('\n')
                for line in lines:
                    if 'AVFoundation video devices' in line or 'iPhone' in line or 'Continuity' in line:
                        print(f"  {line.strip()}")
                        if 'iPhone' in line or 'Continuity' in line:
                            # Extract device index
                            import re
                            match = re.search(r'\[(\d+)\]', line)
                            if match:
                                self.available_devices.append(int(match.group(1)))
            
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            print("⚠️ FFmpeg not available, trying OpenCV device enumeration...")
            self._enumerate_opencv_devices()
    
    def _enumerate_opencv_devices(self):
        """Enumerate devices using OpenCV"""
        print("📱 Enumerating OpenCV video devices...")
        
        for i in range(10):  # Check first 10 device indices
            cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"  Device {i}: Available (Resolution: {frame.shape[1]}x{frame.shape[0]})")
                    self.available_devices.append(i)
                cap.release()
        
        if not self.available_devices:
            print("❌ No video devices found")
        else:
            print(f"✅ Found {len(self.available_devices)} video device(s)")
    
    def connect_iphone_camera(self, device_index=None, resolution=(1280, 720)):
        """Connect to iPhone camera via Continuity Camera"""
        print("📱 Connecting to iPhone camera...")
        
        if device_index is None:
            # Try to find iPhone/Continuity Camera automatically
            device_index = self._find_iphone_device()
        
        if device_index is None:
            print("❌ iPhone camera not found. Please ensure:")
            print("   - iPhone and Mac are signed in with same Apple ID")
            print("   - Wi-Fi and Bluetooth are enabled on both devices")
            print("   - Continuity Camera is enabled in System Preferences")
            print("   - iPhone is nearby and unlocked")
            return False
        
        # Open camera with AVFoundation backend
        self.cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
        
        if not self.cap.isOpened():
            print(f"❌ Failed to open device {device_index}")
            return False
        
        # Set resolution (720p default for better FPS)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Set other properties for better performance
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Verify connection with warm-up retries (Continuity cameras can need a moment)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            import time as _t
            ok = False
            for _ in range(30):  # ~3 seconds total
                _t.sleep(0.1)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    ok = True
                    break
            if not ok:
                print("❌ Failed to read frame from iPhone camera")
                self.cap.release()
                return False
        
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ iPhone camera connected successfully!")
        print(f"   Resolution: {actual_width}x{actual_height}")
        print(f"   FPS: {fps}")
        print(f"   Device index: {device_index}")
        
        self.current_device = device_index
        return True

    def switch_device(self, new_device_index, resolution=(1280, 720)):
        """Switch to a different camera device at runtime"""
        try:
            if self.cap is not None:
                self.cap.release()
            self.current_device = None
            return self.connect_iphone_camera(new_device_index, resolution)
        except Exception:
            return False
    
    def _find_iphone_device(self):
        """Find iPhone/Continuity Camera device automatically"""
        if not self.available_devices:
            self.list_devices()
        
        # Try to find iPhone by name (this is heuristic)
        for device_idx in self.available_devices:
            cap = cv2.VideoCapture(device_idx, cv2.CAP_AVFOUNDATION)
            if cap.isOpened():
                # Try to get device name (not always available)
                device_name = cap.getBackendName()
                cap.release()
                
                # Check if this looks like an iPhone device
                if device_idx > 0:  # Usually iPhone is not device 0
                    print(f"🎯 Selected device {device_idx} as potential iPhone camera")
                    return device_idx
        
        # If no specific iPhone found, use the first available device
        if self.available_devices:
            print(f"🎯 Using device {self.available_devices[0]} as fallback")
            return self.available_devices[0]
        
        return None
    
    def read_frame(self):
        """Read frame from iPhone camera"""
        if self.cap is None or not self.cap.isOpened():
            return None, None
        
        ret, frame = self.cap.read()
        if not ret:
            return None, None
        
        return frame, time.time()
    
    def release(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        print("📱 iPhone camera released")

class PlantStressDetector:
    """Real-time plant stress detection system"""
    
    def __init__(self, yolo_model_path, mobilevit_model_path, class_names=None):
        self.yolo_model = None
        self.mobilevit_model = None
        self.class_names = class_names or []
        # Prefer Apple MPS, then CUDA, then CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        # Load models
        self._load_models(yolo_model_path, mobilevit_model_path)
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.processing_times = deque(maxlen=30)
        
        # Visualization settings
        self.colors = self._generate_colors()
        self.font = self._load_font()
        # Minimal runtime: no GPS, no CSV logging
    
    def _load_models(self, yolo_model_path, mobilevit_model_path):
        """Load YOLO and MobileViT models"""
        print("🤖 Loading models...")
        
        # Load YOLO model
        if yolo_model_path and Path(yolo_model_path).exists():
            self.yolo_model = YOLO(yolo_model_path)
            print(f"✅ YOLO model loaded: {yolo_model_path}")
        else:
            print("⚠️ YOLO model not found, using default YOLO model")
            self.yolo_model = YOLO('yolo11n-seg.pt')

        # Move YOLO to device if supported
        try:
            if hasattr(self.yolo_model, 'to'):
                self.yolo_model.to(str(self.device))
        except Exception:
            pass
        
        # Load MobileViT model
        if mobilevit_model_path and Path(mobilevit_model_path).exists():
            self._load_mobilevit_model(mobilevit_model_path)
        else:
            print("⚠️ MobileViT model not found, classification disabled")
    
    def _load_mobilevit_model(self, model_path):
        """Load MobileViT classification model"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                model_state = checkpoint['model_state_dict']
                config = checkpoint.get('config', {})
                classes = checkpoint.get('classes', self.class_names)
            else:
                model_state = checkpoint.state_dict()
                config = {}
                classes = self.class_names
            
            # Create model architecture
            num_classes = len(classes) if classes else 27  # Default for PlantDoc
            model_name = config.get('model_name', 'mobilevit_xxs.cvnets_in1k')
            
            self.mobilevit_model = timm.create_model(
                model_name, 
                pretrained=False, 
                num_classes=num_classes
            )
            self.mobilevit_model.load_state_dict(model_state)
            self.mobilevit_model = self.mobilevit_model.to(self.device)
            self.mobilevit_model.eval()
            
            self.class_names = classes
            print(f"✅ MobileViT model loaded: {model_path}")
            print(f"   Classes: {len(self.class_names)}")
            
        except Exception as e:
            print(f"❌ Failed to load MobileViT model: {e}")
            self.mobilevit_model = None
    
    def _generate_colors(self):
        """Generate colors for visualization"""
        np.random.seed(42)  # For consistent colors
        colors = []
        for i in range(50):  # Generate colors for up to 50 classes
            color = tuple(map(int, np.random.randint(0, 255, 3)))
            colors.append(color)
        return colors
    
    def _load_font(self):
        """Load font for text overlay"""
        try:
            # Try to load a system font
            font_paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Arial.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            ]
            
            for font_path in font_paths:
                if Path(font_path).exists():
                    return ImageFont.truetype(font_path, 16)
            
            # Fallback to default font
            return ImageFont.load_default()
            
        except Exception:
            return ImageFont.load_default()
    
    def preprocess_frame(self, frame, apply_gray_world=False):
        """Preprocess frame for inference"""
        if apply_gray_world:
            frame = self._apply_gray_world_correction(frame)
        
        return frame
    
    def _apply_gray_world_correction(self, frame):
        """Apply Gray-World color constancy correction"""
        # Convert to float
        frame_float = frame.astype(np.float32) / 255.0
        
        # Calculate mean for each channel
        mean_r = np.mean(frame_float[:, :, 0])
        mean_g = np.mean(frame_float[:, :, 1])
        mean_b = np.mean(frame_float[:, :, 2])
        
        # Calculate scaling factors
        mean_gray = (mean_r + mean_g + mean_b) / 3.0
        scale_r = mean_gray / mean_r if mean_r > 0 else 1.0
        scale_g = mean_gray / mean_g if mean_g > 0 else 1.0
        scale_b = mean_gray / mean_b if mean_b > 0 else 1.0
        
        # Apply correction
        frame_corrected = frame_float.copy()
        frame_corrected[:, :, 0] *= scale_r
        frame_corrected[:, :, 1] *= scale_g
        frame_corrected[:, :, 2] *= scale_b
        
        # Clip to valid range and convert back
        frame_corrected = np.clip(frame_corrected, 0, 1)
        frame_corrected = (frame_corrected * 255).astype(np.uint8)
        
        return frame_corrected
    
    def detect_plants(self, frame):
        """Detect plants using YOLO segmentation"""
        if self.yolo_model is None:
            return []
        
        try:
            # Run YOLO inference at smaller imgsz for higher FPS
            results = self.yolo_model(frame, imgsz=640, verbose=False)
            
            detections = []
            for result in results:
                if result.masks is not None:
                    # Process segmentation masks
                    for i, (mask, box, conf, cls) in enumerate(zip(
                        result.masks.data, result.boxes.xyxy, 
                        result.boxes.conf, result.boxes.cls
                    )):
                        detections.append({
                            'type': 'segmentation',
                            'mask': mask.cpu().numpy(),
                            'box': box.cpu().numpy(),
                            'confidence': conf.cpu().item(),
                            'class_id': int(cls.cpu().item()),
                            'class_name': result.names[int(cls.cpu().item())]
                        })
                elif result.boxes is not None:
                    # Process bounding boxes
                    for i, (box, conf, cls) in enumerate(zip(
                        result.boxes.xyxy, result.boxes.conf, result.boxes.cls
                    )):
                        detections.append({
                            'type': 'detection',
                            'box': box.cpu().numpy(),
                            'confidence': conf.cpu().item(),
                            'class_id': int(cls.cpu().item()),
                            'class_name': result.names[int(cls.cpu().item())]
                        })
            
            return detections
            
        except Exception as e:
            print(f"❌ YOLO inference error: {e}")
            return []
    
    def classify_crop(self, frame, detection):
        """Classify plant stress using MobileViT"""
        if self.mobilevit_model is None:
            return None
        
        try:
            # Extract crop based on detection type
            if detection['type'] == 'segmentation':
                crop = self._extract_mask_crop(frame, detection['mask'])
            else:
                crop = self._extract_box_crop(frame, detection['box'])
            
            if crop is None or crop.size == 0:
                return None
            
            # Preprocess for MobileViT
            crop_tensor = self._preprocess_crop(crop)
            
            # Run inference
            with torch.no_grad():
                output = self.mobilevit_model(crop_tensor.unsqueeze(0).to(self.device))
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # Get class name
            class_name = self.class_names[predicted_class] if predicted_class < len(self.class_names) else f"Class {predicted_class}"
            
            return {
                'class_name': class_name,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy()
            }
            
        except Exception as e:
            print(f"❌ MobileViT inference error: {e}")
            return None
    
    def _extract_mask_crop(self, frame, mask):
        """Extract crop using segmentation mask"""
        # Convert mask to binary
        mask_binary = (mask > 0.5).astype(np.uint8)
        
        # Find bounding box of mask
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(frame.shape[1] - x, w + 2 * padding)
        h = min(frame.shape[0] - y, h + 2 * padding)
        
        return frame[y:y+h, x:x+w]
    
    def _extract_box_crop(self, frame, box):
        """Extract crop using bounding box"""
        x1, y1, x2, y2 = map(int, box)
        
        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        return frame[y1:y2, x1:x2]
    
    def _preprocess_crop(self, crop):
        """Preprocess crop for MobileViT"""
        # Resize to 224x224
        crop_resized = cv2.resize(crop, (224, 224))
        
        # Convert to PIL Image
        crop_pil = Image.fromarray(cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB))
        
        # Apply transforms
        transform = create_transform(
            input_size=(224, 224),
            is_training=False,
            interpolation='bicubic',
        )
        
        return transform(crop_pil)
    
    def visualize_results(self, frame, detections, classifications):
        """Visualize detection and classification results"""
        # Convert to PIL for text overlay
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        for i, detection in enumerate(detections):
            # Get color
            color = self.colors[detection['class_id'] % len(self.colors)]
            
            if detection['type'] == 'segmentation':
                # Draw segmentation mask
                mask = detection['mask']
                # Resize mask to match frame dimensions
                mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                mask_colored = np.zeros_like(frame)
                mask_colored[mask_resized > 0.5] = color
                
                # Blend with original frame
                alpha = 0.3
                frame = cv2.addWeighted(frame, 1 - alpha, mask_colored, alpha, 0)
                
                # Draw contour
                mask_binary = (mask > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    cv2.drawContours(frame, contours, -1, color, 2)
            
            else:
                # Draw bounding box
                x1, y1, x2, y2 = map(int, detection['box'])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Add classification result
            if i < len(classifications) and classifications[i] is not None:
                class_info = classifications[i]
                text = f"{class_info['class_name']}: {class_info['confidence']:.2f}"
                
                # Convert back to PIL for text
                frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(frame_pil)
                
                # Draw text background
                bbox = draw.textbbox((x1, y1 - 20), text, font=self.font)
                draw.rectangle(bbox, fill=(0, 0, 0, 128))
                draw.text((x1, y1 - 20), text, fill=color, font=self.font)
                
                # Convert back to OpenCV format
                frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        
        return frame
    
    def add_performance_overlay(self, frame, fps, processing_time):
        """Add performance metrics overlay"""
        # Update performance tracking
        self.fps_history.append(fps)
        self.processing_times.append(processing_time)
        
        # Calculate averages
        avg_fps = np.mean(self.fps_history) if self.fps_history else 0
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0
        
        # Add text overlay
        overlay_text = [
            f"FPS: {fps:.1f} (avg: {avg_fps:.1f})",
            f"Proc: {processing_time*1000:.1f}ms (avg: {avg_processing_time*1000:.1f}ms)",
            f"Res: {frame.shape[1]}x{frame.shape[0]}"
        ]
        
        y_offset = 30
        for text in overlay_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (0, 255, 0), 2)
            y_offset += 25
        
        return frame

class RealTimeInferenceSystem:
    """Main real-time inference system"""
    
    def __init__(self, yolo_model_path, mobilevit_model_path, class_names=None):
        self.camera_manager = ContinuityCameraManager()
        self.detector = PlantStressDetector(yolo_model_path, mobilevit_model_path, class_names)
        
        # Performance settings
        self.frame_skip = 0  # Process every Nth frame
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.fps = 0
        
        # Threading for performance
        self.running = False
        self.processing_thread = None
        self.latest_results = None
        self.window_name = 'Plant Stress Detection - iPhone Camera'
        
        # GPS & logging
        self.gps_manager = None
        self.logger = None
        self.frame_id = 0
    
    def start(self, device_index=None, resolution=(1280, 720), frame_skip=0):
        """Start real-time inference"""
        print("🚀 Starting real-time plant stress detection...")
        
        # Connect to iPhone camera
        if not self.camera_manager.connect_iphone_camera(device_index, resolution):
            print("❌ Failed to connect to iPhone camera")
            return False
        
        # No GPS/CSV setup in minimal runtime

        self.frame_skip = frame_skip
        self.resolution = resolution
        self.running = True
        
        print("✅ Real-time inference started!")
        print("   Controls: 'q' or ESC to quit, 's' to save frame, 'f' to toggle frame skip, 'c' to cycle camera")
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n⏹️ Stopping real-time inference...")
        finally:
            self.stop()
        
        return True
    
    def _main_loop(self):
        """Main processing loop"""
        while self.running:
            # Read frame
            frame, timestamp = self.camera_manager.read_frame()
            if frame is None:
                print("❌ Failed to read frame")
                time.sleep(0.1)
                continue
            
            # Frame skipping for performance
            self.frame_count += 1
            if self.frame_skip > 0 and self.frame_count % (self.frame_skip + 1) != 0:
                continue
            
            # Process frame
            start_time = time.time()
            
            # Preprocess
            frame_processed = self.detector.preprocess_frame(frame, apply_gray_world=True)
            
            # Detect plants
            detections = self.detector.detect_plants(frame_processed)
            
            # Classify each detection
            classifications = []
            for detection in detections:
                classification = self.detector.classify_crop(frame_processed, detection)
                classifications.append(classification)
            
            processing_time = time.time() - start_time
            # Frame counter not required
            
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_fps_time >= 1.0:
                self.fps = 1.0 / (current_time - self.last_fps_time)
                self.last_fps_time = current_time
            
            # Visualize results
            frame_result = self.detector.visualize_results(frame, detections, classifications)
            frame_result = self.detector.add_performance_overlay(frame_result, self.fps, processing_time)

            # No logging in minimal runtime
            
            # Display results
            cv2.imshow(self.window_name, frame_result)

            # Handle window close
            try:
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except Exception:
                pass
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord('s'):
                self._save_frame(frame_result, detections, classifications)
            elif key == ord('f'):
                self.frame_skip = (self.frame_skip + 1) % 5
                print(f"Frame skip: {self.frame_skip}")
            elif key == ord('c'):
                # cycle between device 0 and 1 quickly; extend as needed
                next_idx = 0 if (self.camera_manager.current_device or 0) != 0 else 1
                if self.camera_manager.switch_device(next_idx, self.resolution):
                    print(f"Switched camera to device {next_idx}")
                else:
                    print("Failed to switch camera")
    
    def _save_frame(self, frame, detections, classifications):
        """Save current frame with results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plant_stress_detection_{timestamp}.jpg"
        
        # Save image
        cv2.imwrite(filename, frame)
        
        # Save metadata
        metadata = {
            'timestamp': timestamp,
            'detections': len(detections),
            'classifications': len(classifications),
            'fps': self.fps,
            'detection_details': []
        }
        
        for i, (detection, classification) in enumerate(zip(detections, classifications)):
            metadata['detection_details'].append({
                'detection': detection,
                'classification': classification
            })
        
        with open(f"plant_stress_detection_{timestamp}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Frame saved: {filename}")
    
    def stop(self):
        """Stop real-time inference"""
        self.running = False
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self.camera_manager.release()
        print("✅ Real-time inference stopped")

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Real-time Plant Stress Detection")
    parser.add_argument("--yolo-model", default="runs/segment/train/weights/best.pt",
                       help="Path to YOLO model")
    parser.add_argument("--mobilevit-model", default="runs/classify/mobilevit_plant_stress_*/best_model.pth",
                       help="Path to MobileViT model")
    parser.add_argument("--class-names", nargs="+", help="Class names for classification")
    parser.add_argument("--device-index", type=int, help="Camera device index")
    parser.add_argument("--resolution", nargs=2, type=int, default=[1920, 1080],
                       help="Camera resolution (width height)")
    parser.add_argument("--frame-skip", type=int, default=0,
                       help="Process every Nth frame (0 = process all frames)")
    parser.add_argument("--list-devices", action="store_true",
                       help="List available video devices and exit")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 6")
    print("Real-time Inference with iPhone Camera")
    print("=" * 60)
    
    # List devices if requested
    if args.list_devices:
        camera_manager = ContinuityCameraManager()
        camera_manager.list_devices()
        return
    
    # Check if models exist
    yolo_model_path = args.yolo_model
    mobilevit_model_path = args.mobilevit_model
    
    if not Path(yolo_model_path).exists():
        print(f"⚠️ YOLO model not found: {yolo_model_path}")
        print("   Using default YOLO model")
        yolo_model_path = None
    
    if not Path(mobilevit_model_path).exists():
        print(f"⚠️ MobileViT model not found: {mobilevit_model_path}")
        print("   Classification will be disabled")
        mobilevit_model_path = None
    
    # Initialize system
    system = RealTimeInferenceSystem(yolo_model_path, mobilevit_model_path, args.class_names)
    
    # Start real-time inference
    success = system.start(
        device_index=args.device_index,
        resolution=tuple(args.resolution),
        frame_skip=args.frame_skip
    )
    
    if not success:
        print("❌ Failed to start real-time inference")
        sys.exit(1)

if __name__ == "__main__":
    main()
