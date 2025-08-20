#!/usr/bin/env python3
"""
Plant Stress Detection Launcher
Simple script to run the plant stress detector
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.inference.plant_stress_detector import main

if __name__ == "__main__":
    main()
