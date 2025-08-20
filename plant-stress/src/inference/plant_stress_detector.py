#!/usr/bin/env python3
"""
Stable MVP for Plant Stress Detection
Fixed version that handles camera issues better
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import timm
from timm.data import create_transform
import argparse
import time

# Class names from the PlantDoc dataset
CLASS_NAMES = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf', 
    'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 
    'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf early blight', 
    'Potato leaf late blight', 'Raspberry leaf', 'Soyabean leaf', 
    'Squash Powdery mildew leaf', 'Strawberry leaf', 'Tomato Early blight leaf', 
    'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 
    'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 
    'Tomato mold leaf', 'grape leaf', 'grape leaf black rot'
]

class StablePlantDetector:
    def __init__(self, model_path, device="auto"):
        """Initialize the plant stress detector"""
        self.device = self._get_device(device)
        print(f"🌱 Using device: {self.device}")
        
        # Create model
        self.model = timm.create_model(
            'mobilevit_xxs.cvnets_in1k',
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )
        
        # Load trained weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Create transforms
        self.transform = create_transform(
            input_size=224,
            is_training=False,
            crop_pct=0.875,
            interpolation='bicubic'
        )
        
        print(f"✅ Model loaded successfully!")
        print(f"📊 Classes: {len(CLASS_NAMES)}")
    
    def _get_device(self, device):
        """Get the best available device"""
        if device == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            elif torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def preprocess_image(self, image):
        """Preprocess image for model inference"""
        try:
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL
            pil_image = Image.fromarray(image_rgb)
            
            # Apply transforms
            tensor = self.transform(pil_image).unsqueeze(0)
            
            return tensor.to(self.device)
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            return None
    
    def predict(self, image):
        """Predict plant stress from image"""
        try:
            # Preprocess
            tensor = self.preprocess_image(image)
            if tensor is None:
                return []
            
            # Inference
            with torch.no_grad():
                outputs = self.model(tensor)
                probabilities = F.softmax(outputs, dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probabilities, 3)
            
            predictions = []
            for i in range(3):
                class_idx = top_indices[0][i].item()
                prob = top_probs[0][i].item()
                class_name = CLASS_NAMES[class_idx]
                predictions.append({
                    'class': class_name,
                    'probability': prob,
                    'confidence': f"{prob:.1%}"
                })
            
            return predictions
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")
            return []
    
    def draw_predictions(self, image, predictions):
        """Draw predictions on image"""
        try:
            # Create overlay
            overlay = image.copy()
            
            # Draw background rectangle
            cv2.rectangle(overlay, (10, 10), (450, 130), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (450, 130), (255, 255, 255), 2)
            
            # Draw title
            cv2.putText(overlay, "Plant Stress Detection", (20, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw predictions
            y_offset = 60
            for i, pred in enumerate(predictions):
                # Color based on confidence
                if pred['probability'] > 0.7:
                    color = (0, 255, 0)  # Green for high confidence
                elif pred['probability'] > 0.4:
                    color = (0, 255, 255)  # Yellow for medium confidence
                else:
                    color = (0, 165, 255)  # Orange for low confidence
                
                # Truncate long class names
                class_name = pred['class']
                if len(class_name) > 25:
                    class_name = class_name[:22] + "..."
                
                text = f"{i+1}. {class_name}: {pred['confidence']}"
                cv2.putText(overlay, text, (20, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_offset += 20
            
            return overlay
        except Exception as e:
            print(f"⚠️ Drawing error: {e}")
            return image

def find_working_camera():
    """Find a working camera device"""
    print("🔍 Searching for working camera...")
    
    for i in range(5):  # Try cameras 0-4
        print(f"  Trying camera {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Camera {i} works!")
                cap.release()
                return i
            cap.release()
    
    print("❌ No working camera found!")
    return None

def main():
    parser = argparse.ArgumentParser(description="Stable Plant Stress Detection MVP")
    parser.add_argument("--model-path", type=str, 
                       default="models/runs/classify/plantdoc_mobilevit_20250820_124432/best_model.pth",
                       help="Path to trained model")
    parser.add_argument("--camera", type=int, default=None, help="Camera device index")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Initialize detector
    print("🚀 Initializing Plant Stress Detector...")
    detector = StablePlantDetector(args.model_path, args.device)
    
    # Find working camera
    camera_index = args.camera if args.camera is not None else find_working_camera()
    if camera_index is None:
        print("❌ No camera available!")
        return
    
    # Open camera
    print(f"📱 Opening camera {camera_index}...")
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print("❌ Failed to open camera!")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Create window
    cv2.namedWindow("Plant Stress Detection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Plant Stress Detection", 1280, 720)
    
    print("🎯 Press 'q' to quit, 's' to save image")
    print("📱 Point camera at plant leaves for stress detection")
    
    frame_count = 0
    last_predictions = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("⚠️ Failed to read frame, retrying...")
                time.sleep(0.1)
                continue
            
            # Process every 5th frame for performance
            if frame_count % 5 == 0:
                try:
                    predictions = detector.predict(frame)
                    if predictions:
                        last_predictions = predictions
                except Exception as e:
                    print(f"⚠️ Prediction error: {e}")
            
            # Draw predictions
            if last_predictions:
                frame_with_predictions = detector.draw_predictions(frame, last_predictions)
            else:
                frame_with_predictions = frame
            
            # Display frame
            cv2.imshow("Plant Stress Detection", frame_with_predictions)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame with predictions
                timestamp = int(time.time())
                filename = f"plant_stress_{timestamp}.jpg"
                cv2.imwrite(filename, frame_with_predictions)
                print(f"💾 Saved image: {filename}")
            
            frame_count += 1
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("👋 Plant Stress Detection stopped")

if __name__ == "__main__":
    main()
