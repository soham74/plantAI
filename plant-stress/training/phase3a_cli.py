#!/usr/bin/env python3
"""
Phase 3A CLI: YOLO Detector/Segmenter Training
Provides the exact command format as specified in the requirements
"""

import subprocess
import sys
import argparse
from pathlib import Path

def run_yolo_training(data_yaml, model_size="n", img_size=512, epochs=100, 
                     batch_size=16, device="auto", project="runs/segment"):
    """
    Run YOLO training with the exact command format specified
    
    Command format:
    yolo task=segment mode=train model=yolo11n-seg.pt data=labels/plantdoc_seg.yaml \
         imgsz=512 epochs=100 batch=16 amp=True workers=8
    """
    
    # Construct model name
    model_name = f"yolo11{model_size}-seg.pt"
    
    # Build command exactly as specified
    cmd = [
        "yolo", "task=segment", "mode=train",
        f"model={model_name}",
        f"data={data_yaml}",
        f"imgsz={img_size}",
        f"epochs={epochs}",
        f"batch={batch_size}",
        "amp=True",
        "workers=8"
    ]
    
    # Add device if specified
    if device != "auto":
        cmd.append(f"device={device}")
    
    # Add project directory if specified
    if project:
        cmd.extend([f"project={project}"])
    
    print("🚀 Running YOLO training with Phase 3A specifications")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ YOLO training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ YOLO training failed: {e}")
        return False

def run_validation(model_path):
    """
    Run YOLO validation
    
    Command format:
    yolo mode=val model=runs/segment/train/weights/best.pt
    """
    
    cmd = ["yolo", "mode=val", f"model={model_path}"]
    
    print(f"🔍 Running model validation")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print("✅ Model validation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model validation failed: {e}")
        return False

def run_export(model_path, format="onnx", imgsz=512):
    """
    Run YOLO export
    
    Command format:
    yolo mode=export model=runs/segment/train/weights/best.pt format=onnx
    """
    
    cmd = [
        "yolo", "mode=export", 
        f"model={model_path}", 
        f"format={format}",
        f"imgsz={imgsz}"
    ]
    
    print(f"📦 Running model export to {format}")
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ Model export to {format} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Model export failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Phase 3A: YOLO CLI Training")
    parser.add_argument("--data-yaml", default="labels/plantdoc_seg.yaml", 
                       help="Data YAML file")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"], 
                       help="YOLO model size")
    parser.add_argument("--img-size", type=int, default=512, choices=[512, 640], 
                       help="Input image size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--project", default="runs/segment", help="Project directory")
    parser.add_argument("--mode", choices=["train", "val", "export"], default="train",
                       help="Mode to run")
    parser.add_argument("--model-path", help="Model path for validation/export")
    parser.add_argument("--export-format", default="onnx", help="Export format")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 3A CLI")
    print("YOLO Detector/Segmenter Training")
    print("=" * 60)
    
    if args.mode == "train":
        print("🎯 Training Mode")
        print("-" * 40)
        success = run_yolo_training(
            args.data_yaml, args.model_size, args.img_size,
            args.epochs, args.batch_size, args.device, args.project
        )
        
        if success:
            print("\n✅ Training completed!")
            print("Next steps:")
            print("1. Validate model: python phase3a_cli.py --mode val --model-path runs/segment/train/weights/best.pt")
            print("2. Export model: python phase3a_cli.py --mode export --model-path runs/segment/train/weights/best.pt")
    
    elif args.mode == "val":
        if not args.model_path:
            print("❌ Model path required for validation mode")
            return
        
        print("🔍 Validation Mode")
        print("-" * 40)
        run_validation(args.model_path)
    
    elif args.mode == "export":
        if not args.model_path:
            print("❌ Model path required for export mode")
            return
        
        print("📦 Export Mode")
        print("-" * 40)
        run_export(args.model_path, args.export_format, args.img_size)

if __name__ == "__main__":
    main()
