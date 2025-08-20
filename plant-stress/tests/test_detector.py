#!/usr/bin/env python3
"""
Tests for Plant Stress Detection System
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from src.inference.plant_stress_detector import StablePlantDetector
from src.utils.model_tester import test_model

class TestPlantStressDetector(unittest.TestCase):
    """Test cases for plant stress detector"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model_path = "models/runs/classify/plantdoc_mobilevit_20250820_124432/best_model.pth"
        
    def test_model_loading(self):
        """Test if model loads correctly"""
        try:
            detector = StablePlantDetector(self.model_path)
            self.assertIsNotNone(detector.model)
            print("✅ Model loading test passed")
        except Exception as e:
            self.fail(f"Model loading failed: {e}")
    
    def test_prediction_shape(self):
        """Test if predictions have correct shape"""
        try:
            detector = StablePlantDetector(self.model_path)
            
            # Create dummy image
            dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Get predictions
            predictions = detector.predict(dummy_image)
            
            # Check predictions
            self.assertIsInstance(predictions, list)
            self.assertGreater(len(predictions), 0)
            self.assertLessEqual(len(predictions), 3)  # Top 3 predictions
            
            # Check prediction structure
            for pred in predictions:
                self.assertIn('class', pred)
                self.assertIn('probability', pred)
                self.assertIn('confidence', pred)
                self.assertIsInstance(pred['probability'], float)
                self.assertGreaterEqual(pred['probability'], 0.0)
                self.assertLessEqual(pred['probability'], 1.0)
            
            print("✅ Prediction shape test passed")
        except Exception as e:
            self.fail(f"Prediction test failed: {e}")
    
    def test_device_selection(self):
        """Test device selection logic"""
        try:
            detector = StablePlantDetector(self.model_path, device="cpu")
            self.assertEqual(detector.device, "cpu")
            print("✅ Device selection test passed")
        except Exception as e:
            self.fail(f"Device selection test failed: {e}")

def run_tests():
    """Run all tests"""
    print("🧪 Running Plant Stress Detection Tests")
    print("=" * 50)
    
    # Run unit tests
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run model test
    print("\n🔍 Testing model functionality...")
    test_model()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    run_tests()
