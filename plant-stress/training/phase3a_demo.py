#!/usr/bin/env python3
"""
Phase 3A Demo: YOLO Detector/Segmenter Training
Demonstrates the exact commands as specified in the requirements
"""

import subprocess
import sys
from pathlib import Path

def demo_phase3a_commands():
    """Demonstrate Phase 3A commands exactly as specified"""
    
    print("🌱 Plant Stress Detection - Phase 3A Demo")
    print("YOLO Detector/Segmenter Training")
    print("=" * 60)
    print()
    
    # Step 1: Training Command (exactly as specified)
    print("🎯 Step 1: Training Command")
    print("-" * 40)
    print("Command (as specified in requirements):")
    print("yolo task=segment mode=train model=yolo11n-seg.pt data=labels/plantdoc_seg.yaml \\")
    print("     imgsz=512 epochs=100 batch=16 amp=True workers=8")
    print()
    
    # Check if data YAML exists
    data_yaml = "labels/plantdoc_seg.yaml"
    if Path(data_yaml).exists():
        print(f"✅ Data YAML found: {data_yaml}")
    else:
        print(f"❌ Data YAML not found: {data_yaml}")
        print("Please ensure the data YAML file exists before running training.")
        return
    
    # Step 2: Validation Command (exactly as specified)
    print("\n🎯 Step 2: Validation Command")
    print("-" * 40)
    print("Command (as specified in requirements):")
    print("yolo mode=val model=runs/segment/train/weights/best.pt")
    print()
    
    # Step 3: Export Command (exactly as specified)
    print("\n🎯 Step 3: Export Command")
    print("-" * 40)
    print("Command (as specified in requirements):")
    print("yolo mode=export model=runs/segment/train/weights/best.pt format=onnx")
    print()
    
    # Step 4: Show how to run with our CLI wrapper
    print("\n🎯 Step 4: Using Our CLI Wrapper")
    print("-" * 40)
    print("Training:")
    print("python training/phase3a_cli.py --mode train")
    print()
    print("Validation:")
    print("python training/phase3a_cli.py --mode val --model-path runs/segment/train/weights/best.pt")
    print()
    print("Export:")
    print("python training/phase3a_cli.py --mode export --model-path runs/segment/train/weights/best.pt")
    print()
    
    # Step 5: Show comprehensive training script
    print("\n🎯 Step 5: Using Comprehensive Training Script")
    print("-" * 40)
    print("Complete pipeline (training + validation + export):")
    print("python training/phase3a_yolo_training.py")
    print()
    print("Training only:")
    print("python training/phase3a_yolo_training.py --skip-validation --skip-export")
    print()
    print("Custom parameters:")
    print("python training/phase3a_yolo_training.py --model-size s --img-size 640 --epochs 150")
    print()
    
    # Step 6: Show Phase 3A specifications
    print("\n🎯 Phase 3A Specifications")
    print("-" * 40)
    print("✅ Task: segment (segmentation)")
    print("✅ Model: yolo11n-seg.pt (nano size)")
    print("✅ Image sizes: 512 or 640")
    print("✅ Augmentation: mosaic, hsv, flips, blur")
    print("✅ Loss components: box, cls, mask")
    print("✅ Optimization: amp=True, workers=8")
    print("✅ Export formats: ONNX, etc.")
    print()
    
    # Step 7: Show expected output structure
    print("\n🎯 Expected Output Structure")
    print("-" * 40)
    print("runs/segment/")
    print("└── plant_stress_seg_n_512_YYYYMMDD_HHMMSS/")
    print("    ├── weights/")
    print("    │   ├── best.pt          # Best model")
    print("    │   └── last.pt          # Last checkpoint")
    print("    ├── results.png          # Training curves")
    print("    ├── confusion_matrix.png # Confusion matrix")
    print("    ├── phase3a_config.json  # Training configuration")
    print("    └── phase3a_summary.json # Training summary")
    print()
    
    print("🎉 Phase 3A Demo Complete!")
    print("Ready to run YOLO training with the specified parameters.")

def check_ultralytics_installation():
    """Check if Ultralytics is properly installed"""
    print("🔍 Checking Ultralytics Installation")
    print("-" * 40)
    
    try:
        import ultralytics
        print(f"✅ Ultralytics version: {ultralytics.__version__}")
        
        # Test YOLO command
        result = subprocess.run(["yolo", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ YOLO CLI available")
            print(f"   Version: {result.stdout.strip()}")
        else:
            print("❌ YOLO CLI not available")
            
    except ImportError:
        print("❌ Ultralytics not installed")
        print("Install with: pip install ultralytics")
        return False
    
    return True

def main():
    print("🌱 Plant Stress Detection - Phase 3A Setup Check")
    print("=" * 60)
    print()
    
    # Check installation
    if not check_ultralytics_installation():
        print("\n❌ Please install Ultralytics before proceeding")
        return
    
    print()
    
    # Show demo
    demo_phase3a_commands()

if __name__ == "__main__":
    main()
