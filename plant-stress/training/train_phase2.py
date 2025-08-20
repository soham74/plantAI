#!/usr/bin/env python3
"""
Phase 2 Training: Complete Hybrid CNN + Transformer Pipeline
Stage A: YOLO Segmentation (CNN for real-time localization)
Stage B: MobileViT Classification (Transformer for stress classification)
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_yolo_training(data_yaml, model_size="n", img_size=512, epochs=100, 
                     batch_size=16, device="auto", project="runs/segment"):
    """Run YOLO segmentation training (Stage A)"""
    print("🚀 Stage A: YOLO Segmentation Training")
    print("=" * 50)
    
    cmd = [
        sys.executable, "training/detect_seg/train_yolo_seg.py",
        "--data", data_yaml,
        "--model-size", model_size,
        "--img-size", str(img_size),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", device,
        "--project", project,
        "--validate",
        "--export"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ YOLO training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ YOLO training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_mobilevit_training(data_dir, model="mobilevit_xxs", img_size=224, 
                          epochs=100, batch_size=32, device="auto", 
                          output_dir="runs/classify"):
    """Run MobileViT classification training (Stage B)"""
    print("\n🚀 Stage B: MobileViT Classification Training")
    print("=" * 50)
    
    cmd = [
        sys.executable, "training/classify_pt/train_mobilevit.py",
        "--data-dir", data_dir,
        "--model", model,
        "--img-size", str(img_size),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--device", device,
        "--output-dir", output_dir
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ MobileViT training completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ MobileViT training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_ablation_study(data_dir, models, img_size=224, batch_size=32, 
                      device="auto", output_dir="runs/ablation"):
    """Run ablation study to compare models"""
    print("\n🔬 Ablation Study: Model Comparison")
    print("=" * 50)
    
    cmd = [
        sys.executable, "training/classify_pt/ablation_study.py",
        "--data-dir", data_dir,
        "--img-size", str(img_size),
        "--batch-size", str(batch_size),
        "--device", device,
        "--output-dir", output_dir,
        "--models"
    ] + models
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Ablation study completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Ablation study failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def find_best_models():
    """Find the best trained models"""
    print("\n🔍 Finding Best Models")
    print("=" * 30)
    
    best_models = {}
    
    # Find best YOLO model
    segment_dir = Path("runs/segment")
    if segment_dir.exists():
        yolo_experiments = list(segment_dir.glob("plant_stress_seg_*"))
        if yolo_experiments:
            latest_yolo = max(yolo_experiments, key=lambda x: x.stat().st_mtime)
            best_yolo = latest_yolo / "weights" / "best.pt"
            if best_yolo.exists():
                best_models['yolo'] = str(best_yolo)
                print(f"✅ Best YOLO model: {best_yolo}")
    
    # Find best MobileViT model
    classify_dir = Path("runs/classify")
    if classify_dir.exists():
        vit_experiments = list(classify_dir.glob("mobilevit_plant_stress_*"))
        if vit_experiments:
            latest_vit = max(vit_experiments, key=lambda x: x.stat().st_mtime)
            best_vit = latest_vit / "best_model.pth"
            if best_vit.exists():
                best_models['mobilevit'] = str(best_vit)
                print(f"✅ Best MobileViT model: {best_vit}")
    
    return best_models

def create_model_summary(best_models, output_dir):
    """Create a summary of the trained models"""
    summary = {
        "phase": "Phase 2 - Modeling (CNN + Transformer)",
        "timestamp": datetime.now().isoformat(),
        "models": best_models,
        "architecture": {
            "stage_a": "YOLO Segmentation (CNN for real-time localization)",
            "stage_b": "MobileViT Classification (Transformer for stress classification)"
        },
        "datasets": {
            "plantvillage": "50,271 images, 38 classes (pretraining/baseline)",
            "plantdoc": "2,414 images, 27 classes (field-like images)",
            "fgvc2020": "1,821 images, 4 classes (apple leaves with noise)"
        }
    }
    
    summary_file = Path(output_dir) / "phase2_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📋 Model summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description="Phase 2: Complete Hybrid CNN + Transformer Training")
    
    # General arguments
    parser.add_argument("--data-yaml", default="labels/plantdoc_seg.yaml", help="YOLO data YAML file")
    parser.add_argument("--data-dir", default="data_proc", help="Classification data directory")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--output-dir", default="runs/phase2", help="Output directory")
    
    # YOLO training arguments
    parser.add_argument("--yolo-model-size", default="n", choices=["n", "s", "m", "l", "x"], 
                       help="YOLO model size")
    parser.add_argument("--yolo-img-size", type=int, default=512, help="YOLO image size")
    parser.add_argument("--yolo-epochs", type=int, default=100, help="YOLO training epochs")
    parser.add_argument("--yolo-batch-size", type=int, default=16, help="YOLO batch size")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO training")
    
    # MobileViT training arguments
    parser.add_argument("--vit-model", default="mobilevit_xxs", help="MobileViT model variant")
    parser.add_argument("--vit-img-size", type=int, default=224, help="MobileViT image size")
    parser.add_argument("--vit-epochs", type=int, default=100, help="MobileViT training epochs")
    parser.add_argument("--vit-batch-size", type=int, default=32, help="MobileViT batch size")
    parser.add_argument("--skip-vit", action="store_true", help="Skip MobileViT training")
    
    # Ablation study arguments
    parser.add_argument("--run-ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--ablation-models", nargs="+", default=[
        "mobilevit_xxs", "mobilevit_xs", "mobilevit_s",
        "efficientnet_lite0", "efficientnet_lite1", "efficientnet_lite2",
        "mobilenetv3_small_100", "mobilenetv3_large_100"
    ], help="Models for ablation study")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation study")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 2 Training")
    print("Hybrid CNN + Transformer Pipeline")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"phase2_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['timestamp'] = timestamp
    
    with open(output_dir / "phase2_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    success_count = 0
    total_stages = 0
    
    # Stage A: YOLO Segmentation Training
    if not args.skip_yolo:
        total_stages += 1
        print("🎯 Stage A: YOLO Segmentation Training")
        print("Purpose: Real-time localization of lesions/leaves")
        print(f"Model: YOLO11{args.yolo_model_size}-seg")
        print(f"Image size: {args.yolo_img_size}")
        print(f"Epochs: {args.yolo_epochs}")
        print()
        
        if run_yolo_training(
            args.data_yaml, args.yolo_model_size, args.yolo_img_size,
            args.yolo_epochs, args.yolo_batch_size, args.device
        ):
            success_count += 1
    else:
        print("⏭️ Skipping YOLO training")
    
    # Stage B: MobileViT Classification Training
    if not args.skip_vit:
        total_stages += 1
        print("\n🎯 Stage B: MobileViT Classification Training")
        print("Purpose: Stress classification and severity regression")
        print(f"Model: {args.vit_model}")
        print(f"Image size: {args.vit_img_size}")
        print(f"Epochs: {args.vit_epochs}")
        print()
        
        if run_mobilevit_training(
            args.data_dir, args.vit_model, args.vit_img_size,
            args.vit_epochs, args.vit_batch_size, args.device
        ):
            success_count += 1
    else:
        print("⏭️ Skipping MobileViT training")
    
    # Ablation Study
    if args.run_ablation and not args.skip_ablation:
        total_stages += 1
        print("\n🎯 Ablation Study: Model Comparison")
        print("Purpose: Compare speed/accuracy trade-offs")
        print(f"Models: {args.ablation_models}")
        print()
        
        if run_ablation_study(
            args.data_dir, args.ablation_models, args.vit_img_size,
            args.vit_batch_size, args.device
        ):
            success_count += 1
    else:
        print("⏭️ Skipping ablation study")
    
    # Find best models
    best_models = find_best_models()
    
    # Create summary
    create_model_summary(best_models, output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏁 Phase 2 Training Summary")
    print("=" * 60)
    print(f"Completed stages: {success_count}/{total_stages}")
    
    if best_models:
        print("\n✅ Best Models Found:")
        for model_type, model_path in best_models.items():
            print(f"  {model_type.upper()}: {model_path}")
    else:
        print("\n❌ No trained models found")
    
    print(f"\n📁 All results saved to: {output_dir}")
    
    if success_count == total_stages:
        print("\n🎉 Phase 2 completed successfully!")
        print("Ready for Phase 3: Real-time Processing")
    else:
        print(f"\n⚠️ Phase 2 partially completed ({success_count}/{total_stages} stages)")
        print("Check logs for details on failed stages")

if __name__ == "__main__":
    main()
