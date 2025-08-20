#!/usr/bin/env python3
"""
Model Training Launcher
Simple script to train the plant stress detection model
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.mobilevit_trainer import main

if __name__ == "__main__":
    main()
