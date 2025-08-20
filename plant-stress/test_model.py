#!/usr/bin/env python3
"""
Model Testing Launcher
Simple script to test the plant stress detection model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.model_tester import test_model

if __name__ == "__main__":
    test_model()
