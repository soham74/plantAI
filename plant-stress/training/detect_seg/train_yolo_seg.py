#!/usr/bin/env python3
"""
YOLO Segmentation Training for Plant Stress Detection
Stage A: CNN-based real-time localization of lesions/leaves
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import yaml
import json
from datetime import datetime

def validate_dataset(data_yaml):
    """Validate dataset configuration"""
    print(f"Validating dataset configuration: {data_yaml}")
    
    if not Path(data_yaml).exists():
        print(f"Error: Data YAML file not found: {data_yaml}")
        return False
    
    # Load and validate YAML
    with open(data_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    required_keys = ['path', 'train', 'val', 'nc', 'names']
    for key in required_keys:
        if key not in config:
            print(f"Error: Missing required key '{key}' in {data_yaml}")
            return False
    
    # Check if paths exist
    base_path = Path(config['path'])
    train_path = base_path / config['train']
    val_path = base_path / config['val']
    
    if not train_path.exists():
        print(f"Error: Training path not found: {train_path}")
        return False
    
    if not val_path.exists():
        print(f"Error: Validation path not found: {val_path}")
        return False
    
    print(f"✅ Dataset validation passed:")
    print(f"   Classes: {config['nc']}")
    print(f"   Train path: {train_path}")
    print(f"   Val path: {val_path}")
    
    return True

def train_yolo_segmentation(data_yaml, model_size="n", img_size=512, epochs=100, 
                           batch_size=16, device="auto", project="runs/segment", 
                           name="plant_stress_seg"):
    """
    Train YOLO segmentation model
    
    Args:
        data_yaml: Path to data configuration YAML file
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        img_size: Input image size
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to use ('auto', 'cpu', '0', '1', etc.)
        project: Project directory
        name: Experiment name
    """
    
    # Validate dataset
    if not validate_dataset(data_yaml):
        return False
    
    # Construct model name
    model_name = f"yolo11{model_size}-seg.pt"
    
    # Create timestamp for unique experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{name}_{model_size}_{img_size}_{timestamp}"
    
    # Construct training command with recommended parameters
    cmd = [
        "yolo", "task=segment", "mode=train",
        f"model={model_name}",
        f"data={data_yaml}",
        f"imgsz={img_size}",
        f"epochs={epochs}",
        f"batch={batch_size}",
        f"device={device}",
        f"project={project}",
        f"name={exp_name}",
        
        # Optimization parameters
        "amp=True",  # Automatic mixed precision
        "workers=8",  # Data loading workers
        
        # Learning rate and optimization
        "lr0=0.01",  # Initial learning rate
        "lrf=0.01",  # Final learning rate factor
        "momentum=0.937",  # SGD momentum
        "weight_decay=0.0005",  # Weight decay
        "warmup_epochs=3",  # Warmup epochs
        "warmup_momentum=0.8",  # Warmup momentum
        "warmup_bias_lr=0.1",  # Warmup bias learning rate
        
        # Augmentation parameters
        "hsv_h=0.015",  # HSV-Hue augmentation
        "hsv_s=0.7",    # HSV-Saturation augmentation
        "hsv_v=0.4",    # HSV-Value augmentation
        "degrees=0.0",  # Image rotation
        "translate=0.1",  # Image translation
        "scale=0.5",    # Image scaling
        "shear=0.0",    # Image shear
        "perspective=0.0",  # Perspective transform
        "flipud=0.0",   # Vertical flip
        "fliplr=0.5",   # Horizontal flip
        "mosaic=1.0",   # Mosaic augmentation
        "mixup=0.0",    # Mixup augmentation
        "copy_paste=0.0",  # Copy-paste augmentation
        
        # Loss weights
        "box=7.5",      # Box loss gain
        "cls=0.5",      # Classification loss gain
        "dfl=1.5",      # DFL loss gain
        "pose=12.0",    # Pose loss gain
        "kobj=2.0",     # Keypoint obj loss gain
        "label_smoothing=0.0",  # Label smoothing epsilon
        
        # Validation parameters
        "val=True",     # Validate during training
        "save_period=10",  # Save checkpoint every x epochs
        
        # Logging
        "plots=True",   # Save plots
        "save=True",    # Save checkpoints
        "cache=False",  # Cache images for faster training
    ]
    
    print(f"🚀 Starting YOLO segmentation training...")
    print(f"   Model: {model_name}")
    print(f"   Image size: {img_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print(f"   Project: {project}/{exp_name}")
    print()
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Training completed successfully!")
        
        # Save training configuration
        config_path = Path(project) / exp_name / "training_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        training_config = {
            "model": model_name,
            "data_yaml": data_yaml,
            "img_size": img_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "command": " ".join(cmd),
            "timestamp": timestamp,
            "status": "completed"
        }
        
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False

def validate_trained_model(model_path):
    """Validate trained model"""
    print(f"🔍 Validating model: {model_path}")
    
    cmd = ["yolo", "mode=val", f"model={model_path}"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Validation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Validation failed: {e}")
        return False

def export_model(model_path, format="onnx", imgsz=512, half=True):
    """Export model to different formats"""
    print(f"📦 Exporting model to {format} format...")
    
    cmd = [
        "yolo", "mode=export", 
        f"model={model_path}", 
        f"format={format}",
        f"imgsz={imgsz}",
        f"half={str(half).lower()}"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Export to {format} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Export failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Train YOLO segmentation model for plant stress detection")
    parser.add_argument("--data", default="labels/plantdoc_seg.yaml", help="Path to data YAML file")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], 
                       help="YOLO model size (n=nano, s=small, m=medium, l=large, x=xlarge)")
    parser.add_argument("--img-size", type=int, default=512, help="Input image size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cpu, 0, 1, etc.)")
    parser.add_argument("--project", default="runs/segment", help="Project directory")
    parser.add_argument("--name", default="plant_stress_seg", help="Experiment name")
    parser.add_argument("--validate", action="store_true", help="Validate model after training")
    parser.add_argument("--export", action="store_true", help="Export model after training")
    parser.add_argument("--export-format", default="onnx", help="Export format")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - YOLO Segmentation Training")
    print("=" * 60)
    
    # Train model
    success = train_yolo_segmentation(
        args.data, 
        args.model_size, 
        args.img_size, 
        args.epochs, 
        args.batch_size,
        args.device,
        args.project,
        args.name
    )
    
    if not success:
        print("❌ Training failed!")
        return
    
    # Find best model
    exp_dir = Path(args.project) / f"{args.name}_{args.model_size}_{args.img_size}_*"
    exp_dirs = list(exp_dir.parent.glob(exp_dir.name))
    
    if exp_dirs:
        latest_exp = max(exp_dirs, key=lambda x: x.stat().st_mtime)
        best_model = latest_exp / "weights" / "best.pt"
        
        if best_model.exists():
            print(f"🏆 Best model saved at: {best_model}")
            
            # Validate if requested
            if args.validate:
                validate_trained_model(str(best_model))
            
            # Export if requested
            if args.export:
                export_model(str(best_model), args.export_format, args.img_size)
        else:
            print("❌ Best model not found!")
    else:
        print("❌ Training experiment directory not found!")

if __name__ == "__main__":
    main()
