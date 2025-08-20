#!/usr/bin/env python3
"""
Setup script for Plant Stress Detection System
"""

from setuptools import setup, find_packages

setup(
    name="plant-stress-detection",
    version="1.0.0",
    description="AI system for detecting plant stress and diseases using computer vision",
    author="PlantAI Team",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "ultralytics>=8.3.0",
        "timm>=1.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "tensorboard>=2.13.0",
    ],
    python_requires=">=3.8",
)
