#!/usr/bin/env python3
"""
YOLO training script for plant stress detection
Trains YOLO segmentation model on PlantDoc dataset
"""

import os
import sys
import subprocess
from pathlib import Path
import argparse

def train_yolo_segmentation(data_yaml, model_size="n", img_size=512, epochs=100, batch_size=16):
    """
    Train YOLO segmentation model
    
    Args:
        data_yaml: Path to data configuration YAML file
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        img_size: Input image size
        epochs: Number of training epochs
        batch_size: Batch size
    """
    
    # Construct model name
    model_name = f"yolo11{model_size}-seg.pt"
    
    # Construct training command
    cmd = [
        "yolo", "task=segment", "mode=train",
        f"model={model_name}",
        f"data={data_yaml}",
        f"imgsz={img_size}",
        f"epochs={epochs}",
        f"batch={batch_size}",
        "amp=True",
        "workers=8",
        "lr0=0.01",
        "weight_decay=0.0005",
        "hsv_h=0.015",
        "hsv_s=0.7",
        "hsv_v=0.4",
        "mosaic=1.0"
    ]
    
    print(f"Training YOLO segmentation model with command:")
    print(" ".join(cmd))
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def validate_model(model_path):
    """Validate trained model"""
    cmd = ["yolo", "mode=val", f"model={model_path}"]
    
    print(f"Validating model: {model_path}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Validation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Validation failed with error: {e}")
        return False

def export_model(model_path, format="onnx"):
    """Export model to different formats"""
    cmd = ["yolo", "mode=export", f"model={model_path}", f"format={format}"]
    
    print(f"Exporting model to {format} format...")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Export to {format} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Export failed with error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model for plant stress detection")
    parser.add_argument("--data", default="labels/plantdoc_seg.yaml", help="Path to data YAML file")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], 
                       help="YOLO model size")
    parser.add_argument("--img-size", type=int, default=512, help="Input image size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--validate", action="store_true", help="Validate model after training")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    parser.add_argument("--export-format", default="onnx", help="Export format")
    
    args = parser.parse_args()
    
    # Check if data YAML exists
    if not Path(args.data).exists():
        print(f"Data YAML file not found: {args.data}")
        print("Please run the dataset analysis script first:")
        print("python scripts/analyze_datasets.py")
        return
    
    # Train model
    success = train_yolo_segmentation(
        args.data, 
        args.model_size, 
        args.img_size, 
        args.epochs, 
        args.batch_size
    )
    
    if not success:
        print("Training failed!")
        return
    
    # Find best model
    runs_dir = Path("runs/segment/train")
    if runs_dir.exists():
        best_model = runs_dir / "weights" / "best.pt"
        if best_model.exists():
            print(f"Best model saved at: {best_model}")
            
            # Validate if requested
            if args.validate:
                validate_model(str(best_model))
            
            # Export if requested
            if args.export:
                export_model(str(best_model), args.export_format)
        else:
            print("Best model not found!")
    else:
        print("Training runs directory not found!")

if __name__ == "__main__":
    main()
