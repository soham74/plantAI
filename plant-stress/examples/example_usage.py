#!/usr/bin/env python3
"""
Example Usage of Plant Stress Detection System
Demonstrates how to use the plant stress detection system
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.plant_stress_detector import StablePlantDetector
from src.utils.model_tester import test_model
import cv2
import numpy as np

def example_basic_usage():
    """Example of basic usage"""
    print("🌱 Plant Stress Detection - Basic Usage Example")
    print("=" * 50)
    
    # Test the model first
    print("1. Testing model...")
    test_model()
    
    print("\n2. Example completed!")
    print("💡 Run 'python run_detector.py' to start camera detection")

def example_custom_detection():
    """Example of custom detection on an image"""
    print("🌱 Plant Stress Detection - Custom Detection Example")
    print("=" * 50)
    
    # Initialize detector
    model_path = "models/runs/classify/plantdoc_mobilevit_20250820_124432/best_model.pth"
    detector = StablePlantDetector(model_path)
    
    # Create a dummy image (you can replace this with a real image)
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Get predictions
    predictions = detector.predict(dummy_image)
    
    print("📊 Predictions on dummy image:")
    for i, pred in enumerate(predictions):
        print(f"  {i+1}. {pred['class']}: {pred['confidence']}")
    
    print("\n✅ Custom detection example completed!")

if __name__ == "__main__":
    print("🚀 Plant Stress Detection Examples")
    print("=" * 40)
    
    # Run examples
    example_basic_usage()
    print()
    example_custom_detection()
    
    print("\n🎉 All examples completed!")
    print("💡 Check the README.md for more usage instructions")
