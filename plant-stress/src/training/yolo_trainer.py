#!/usr/bin/env python3
"""
Phase 3A: YOLO Detector/Segmenter Training
Ultralytics YOLO training with comprehensive validation and export
"""

import os
import sys
import subprocess
import argparse
import yaml
import json
from pathlib import Path
from datetime import datetime

def validate_data_yaml(data_yaml):
    """Validate the data YAML configuration"""
    print(f"🔍 Validating data configuration: {data_yaml}")
    
    if not Path(data_yaml).exists():
        print(f"❌ Data YAML file not found: {data_yaml}")
        return False
    
    try:
        with open(data_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        required_keys = ['path', 'train', 'val', 'nc', 'names']
        for key in required_keys:
            if key not in config:
                print(f"❌ Missing required key '{key}' in {data_yaml}")
                return False
        
        # Check if paths exist
        base_path = Path(config['path'])
        train_path = base_path / config['train']
        val_path = base_path / config['val']
        
        if not train_path.exists():
            print(f"❌ Training path not found: {train_path}")
            return False
        
        if not val_path.exists():
            print(f"❌ Validation path not found: {val_path}")
            return False
        
        print(f"✅ Data validation passed:")
        print(f"   Classes: {config['nc']}")
        print(f"   Train path: {train_path}")
        print(f"   Val path: {val_path}")
        print(f"   Class names: {config['names'][:5]}..." if len(config['names']) > 5 else f"   Class names: {config['names']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating data YAML: {e}")
        return False

def train_yolo_segmentation(data_yaml, model_size="n", img_size=512, epochs=100, 
                           batch_size=16, device="auto", project="runs/segment", 
                           name="plant_stress_seg"):
    """
    Train YOLO segmentation model with Phase 3A specifications
    
    Args:
        data_yaml: Path to data configuration YAML file
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        img_size: Input image size (512 or 640)
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to use ('auto', 'cpu', '0', '1', etc.)
        project: Project directory
        name: Experiment name
    """
    
    # Validate data configuration
    if not validate_data_yaml(data_yaml):
        return False
    
    # Construct model name
    model_name = f"yolo11{model_size}-seg.pt"
    
    # Create timestamp for unique experiment name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{name}_{model_size}_{img_size}_{timestamp}"
    
    print(f"🚀 Starting YOLO segmentation training (Phase 3A)")
    print(f"   Model: {model_name}")
    print(f"   Image size: {img_size}")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Device: {device}")
    print(f"   Project: {project}/{exp_name}")
    print()
    
    # Construct training command with Phase 3A specifications
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
        
        # Phase 3A specific parameters
        "amp=True",  # Automatic mixed precision
        "workers=8",  # Data loading workers
        
        # Augmentation parameters (as specified)
        "mosaic=1.0",  # Mosaic augmentation
        "hsv_h=0.015",  # HSV-Hue augmentation
        "hsv_s=0.7",    # HSV-Saturation augmentation
        "hsv_v=0.4",    # HSV-Value augmentation
        "fliplr=0.5",   # Horizontal flip
        "blur=0.01",    # Blur augmentation
        
        # Loss weights for segmentation
        "box=7.5",      # Box loss gain
        "cls=0.5",      # Classification loss gain
        "dfl=1.5",      # DFL loss gain
        "pose=12.0",    # Pose loss gain (for segmentation)
        "kobj=2.0",     # Keypoint obj loss gain
        
        # Optimization parameters
        "lr0=0.01",  # Initial learning rate
        "lrf=0.01",  # Final learning rate factor
        "momentum=0.937",  # SGD momentum
        "weight_decay=0.0005",  # Weight decay
        "warmup_epochs=3",  # Warmup epochs
        "warmup_momentum=0.8",  # Warmup momentum
        "warmup_bias_lr=0.1",  # Warmup bias learning rate
        
        # Additional augmentations
        "degrees=0.0",  # Image rotation
        "translate=0.1",  # Image translation
        "scale=0.5",    # Image scaling
        "shear=0.0",    # Image shear
        "perspective=0.0",  # Perspective transform
        "flipud=0.0",   # Vertical flip
        "mixup=0.0",    # Mixup augmentation
        "copy_paste=0.0",  # Copy-paste augmentation
        
        # Validation and logging
        "val=True",     # Validate during training
        "save_period=10",  # Save checkpoint every x epochs
        "plots=True",   # Save plots
        "save=True",    # Save checkpoints
        "cache=False",  # Cache images for faster training
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print()
    
    # Run training
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ YOLO training completed successfully!")
        
        # Save training configuration
        config_path = Path(project) / exp_name / "phase3a_config.json"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        training_config = {
            "phase": "Phase 3A - YOLO Detector/Segmenter Training",
            "model": model_name,
            "data_yaml": data_yaml,
            "img_size": img_size,
            "epochs": epochs,
            "batch_size": batch_size,
            "device": device,
            "command": " ".join(cmd),
            "timestamp": timestamp,
            "status": "completed",
            "specifications": {
                "augmentation": ["mosaic", "hsv", "flips", "blur"],
                "loss_components": ["box", "cls", "mask"],
                "optimization": ["amp", "workers=8", "mixed_precision"]
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(training_config, f, indent=2)
        
        return True, exp_name
        
    except subprocess.CalledProcessError as e:
        print(f"❌ YOLO training failed with error: {e}")
        print(f"Error output: {e.stderr}")
        return False, None

def validate_trained_model(model_path):
    """Validate trained model using YOLO validation"""
    print(f"🔍 Validating model: {model_path}")
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    cmd = ["yolo", "mode=val", f"model={model_path}"]
    
    print(f"Validation command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Model validation completed successfully!")
        
        # Extract validation metrics from output
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'mAP50' in line or 'mAP50-95' in line:
                print(f"   {line.strip()}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Model validation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def export_model(model_path, format="onnx", imgsz=512, half=True):
    """Export model to different formats"""
    print(f"📦 Exporting model to {format} format...")
    
    if not Path(model_path).exists():
        print(f"❌ Model file not found: {model_path}")
        return False
    
    cmd = [
        "yolo", "mode=export", 
        f"model={model_path}", 
        f"format={format}",
        f"imgsz={imgsz}",
        f"half={str(half).lower()}"
    ]
    
    print(f"Export command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Export to {format} completed successfully!")
        
        # Find exported model
        model_dir = Path(model_path).parent
        exported_files = list(model_dir.glob(f"*.{format}"))
        if exported_files:
            print(f"   Exported model: {exported_files[0]}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Export failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def find_best_model(project_dir, exp_name):
    """Find the best trained model"""
    best_model_path = Path(project_dir) / exp_name / "weights" / "best.pt"
    
    if best_model_path.exists():
        print(f"🏆 Best model found: {best_model_path}")
        return str(best_model_path)
    else:
        print(f"❌ Best model not found at: {best_model_path}")
        return None

def create_training_summary(exp_name, model_path, project_dir):
    """Create a summary of the training results"""
    summary = {
        "phase": "Phase 3A - YOLO Detector/Segmenter Training",
        "experiment": exp_name,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat(),
        "specifications": {
            "task": "segment",
            "augmentation": ["mosaic", "hsv", "flips", "blur"],
            "loss_components": ["box", "cls", "mask"],
            "optimization": ["amp", "workers=8", "mixed_precision"]
        },
        "training_params": {
            "model": "yolo11n-seg.pt",
            "img_size": 512,
            "epochs": 100,
            "batch_size": 16,
            "amp": True,
            "workers": 8
        }
    }
    
    summary_file = Path(project_dir) / exp_name / "phase3a_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Training summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Phase 3A: YOLO Detector/Segmenter Training")
    parser.add_argument("--data-yaml", default="labels/plantdoc_seg.yaml", help="Data YAML file")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], 
                       help="YOLO model size")
    parser.add_argument("--img-size", type=int, default=512, choices=[512, 640], 
                       help="Input image size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--project", default="runs/segment", help="Project directory")
    parser.add_argument("--name", default="plant_stress_seg", help="Experiment name")
    parser.add_argument("--skip-training", action="store_true", help="Skip training")
    parser.add_argument("--skip-validation", action="store_true", help="Skip validation")
    parser.add_argument("--skip-export", action="store_true", help="Skip export")
    parser.add_argument("--export-format", default="onnx", help="Export format")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 3A")
    print("YOLO Detector/Segmenter Training")
    print("=" * 60)
    print(f"Data YAML: {args.data_yaml}")
    print(f"Model: YOLO11{args.model_size}-seg")
    print(f"Image size: {args.img_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {args.device}")
    print()
    
    # Step 1: Training
    if not args.skip_training:
        print("🎯 Step 1: YOLO Segmentation Training")
        print("-" * 40)
        
        success, exp_name = train_yolo_segmentation(
            args.data_yaml, args.model_size, args.img_size,
            args.epochs, args.batch_size, args.device,
            args.project, args.name
        )
        
        if not success:
            print("❌ Training failed!")
            return
    else:
        print("⏭️ Skipping training")
        exp_name = f"{args.name}_{args.model_size}_{args.img_size}_*"
    
    # Step 2: Find best model
    print("\n🔍 Step 2: Finding Best Model")
    print("-" * 40)
    
    # Find the latest experiment if exp_name contains wildcard
    if "*" in exp_name:
        exp_pattern = Path(args.project) / exp_name
        exp_dirs = list(exp_pattern.parent.glob(exp_pattern.name))
        if exp_dirs:
            exp_name = max(exp_dirs, key=lambda x: x.stat().st_mtime).name
        else:
            print("❌ No training experiments found!")
            return
    
    best_model = find_best_model(args.project, exp_name)
    if not best_model:
        print("❌ Best model not found!")
        return
    
    # Step 3: Validation
    if not args.skip_validation:
        print("\n🎯 Step 3: Model Validation")
        print("-" * 40)
        
        validate_trained_model(best_model)
    else:
        print("⏭️ Skipping validation")
    
    # Step 4: Export
    if not args.skip_export:
        print("\n🎯 Step 4: Model Export")
        print("-" * 40)
        
        export_model(best_model, args.export_format, args.img_size)
    else:
        print("⏭️ Skipping export")
    
    # Step 5: Create summary
    print("\n📋 Step 5: Creating Summary")
    print("-" * 40)
    
    create_training_summary(exp_name, best_model, args.project)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 Phase 3A Training Summary")
    print("=" * 60)
    print(f"✅ Training completed: {exp_name}")
    print(f"✅ Best model: {best_model}")
    print(f"✅ Project directory: {args.project}")
    
    if not args.skip_export:
        print(f"✅ Model exported to: {args.export_format}")
    
    print(f"\n🎉 Phase 3A completed successfully!")
    print("Ready for Phase 3B: MobileViT Classification Training")

if __name__ == "__main__":
    main()
