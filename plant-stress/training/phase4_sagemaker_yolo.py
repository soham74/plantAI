#!/usr/bin/env python3
"""
Phase 4: SageMaker YOLO Training Script
YOLO training with SMDDP support for hyperparameter tuning
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """Setup distributed training with SMDDP"""
    if 'SM_HOSTS' in os.environ:
        # SageMaker distributed training
        hosts = json.loads(os.environ['SM_HOSTS'])
        current_host = os.environ['SM_CURRENT_HOST']
        rank = hosts.index(current_host)
        world_size = len(hosts)
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        print(f"✅ SMDDP initialized: rank={rank}, world_size={world_size}")
        return rank, world_size
    else:
        # Local training
        print("⚠️ Running in local mode (no distributed training)")
        return 0, 1

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def download_dataset_from_s3(s3_uri, local_path):
    """Download dataset from S3"""
    print(f"📥 Downloading dataset from {s3_uri}")
    
    # Use AWS CLI for downloading
    cmd = [
        "aws", "s3", "cp", s3_uri, local_path, "--recursive"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ Dataset downloaded to {local_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download dataset: {e}")
        return False

def extract_dataset(archive_path, extract_path):
    """Extract dataset archive"""
    import zipfile
    
    print(f"📦 Extracting dataset from {archive_path}")
    
    try:
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        print(f"✅ Dataset extracted to {extract_path}")
        return True
    except Exception as e:
        print(f"❌ Failed to extract dataset: {e}")
        return False

def train_yolo_with_hyperparameters(hyperparameters):
    """Train YOLO with given hyperparameters"""
    
    # Extract hyperparameters
    model_size = hyperparameters.get('model_size', 'n')
    img_size = hyperparameters.get('img_size', 512)
    epochs = hyperparameters.get('epochs', 100)
    batch_size = hyperparameters.get('batch_size', 16)
    lr0 = hyperparameters.get('lr0', 0.01)
    weight_decay = hyperparameters.get('weight_decay', 0.0005)
    hsv_h = hyperparameters.get('hsv_h', 0.015)
    hsv_s = hyperparameters.get('hsv_s', 0.7)
    hsv_v = hyperparameters.get('hsv_v', 0.4)
    mosaic = hyperparameters.get('mosaic', 1.0)
    fliplr = hyperparameters.get('fliplr', 0.5)
    blur = hyperparameters.get('blur', 0.01)
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    
    try:
        # Construct YOLO training command
        cmd = [
            "yolo", "task=segment", "mode=train",
            f"model=yolo11{model_size}-seg.pt",
            "data=labels/plantdoc_seg.yaml",
            f"imgsz={img_size}",
            f"epochs={epochs}",
            f"batch={batch_size}",
            f"lr0={lr0}",
            f"weight_decay={weight_decay}",
            f"hsv_h={hsv_h}",
            f"hsv_s={hsv_s}",
            f"hsv_v={hsv_v}",
            f"mosaic={mosaic}",
            f"fliplr={fliplr}",
            f"blur={blur}",
            "amp=True",
            "workers=8",
            "device=auto",
            "project=runs/segment",
            "name=sagemaker_yolo_training"
        ]
        
        # Add distributed training parameters
        if world_size > 1:
            cmd.extend([
                f"device={rank}",
                f"world_size={world_size}"
            ])
        
        print(f"🚀 Starting YOLO training with hyperparameters:")
        print(f"   Model: yolo11{model_size}-seg")
        print(f"   Image size: {img_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Batch size: {batch_size}")
        print(f"   Learning rate: {lr0}")
        print(f"   Weight decay: {weight_decay}")
        print(f"   Rank: {rank}, World size: {world_size}")
        print()
        print(f"Command: {' '.join(cmd)}")
        print()
        
        # Run training
        result = subprocess.run(cmd, check=True)
        print("✅ YOLO training completed successfully!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ YOLO training failed: {e}")
        return False
    finally:
        cleanup_distributed()

def evaluate_model(model_path):
    """Evaluate trained model and return metrics"""
    print(f"🔍 Evaluating model: {model_path}")
    
    cmd = ["yolo", "mode=val", f"model={model_path}"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Parse metrics from output
        output = result.stdout
        metrics = {}
        
        # Extract mAP metrics
        for line in output.split('\n'):
            if 'mAP50' in line:
                try:
                    map50 = float(line.split()[-1])
                    metrics['mAP50'] = map50
                except:
                    pass
            elif 'mAP50-95' in line:
                try:
                    map50_95 = float(line.split()[-1])
                    metrics['mAP50-95'] = map50_95
                except:
                    pass
        
        print(f"✅ Model evaluation completed:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Model evaluation failed: {e}")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Phase 4: SageMaker YOLO Training")
    parser.add_argument("--hyperparameters", type=json.loads, default="{}", 
                       help="Hyperparameters as JSON string")
    parser.add_argument("--model-dir", default="/opt/ml/model", 
                       help="Model output directory")
    parser.add_argument("--data-dir", default="/opt/ml/input/data/training", 
                       help="Training data directory")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 4")
    print("SageMaker YOLO Training")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Hyperparameters: {args.hyperparameters}")
    print()
    
    # Check if running in SageMaker
    if 'SM_TRAINING_ENV' in os.environ:
        print("✅ Running in SageMaker environment")
        training_env = json.loads(os.environ['SM_TRAINING_ENV'])
        print(f"   Job name: {training_env.get('job_name', 'N/A')}")
        print(f"   Current host: {training_env.get('current_host', 'N/A')}")
    else:
        print("⚠️ Running in local environment")
    
    # Download dataset if S3 URI is provided
    if 'SM_CHANNEL_TRAINING' in os.environ:
        training_data_path = os.environ['SM_CHANNEL_TRAINING']
        print(f"📁 Training data path: {training_data_path}")
        
        # Check if it's a zip file
        if training_data_path.endswith('.zip'):
            extract_path = "/tmp/dataset"
            if extract_dataset(training_data_path, extract_path):
                os.environ['DATASET_PATH'] = extract_path
            else:
                print("❌ Failed to extract dataset")
                return
        else:
            os.environ['DATASET_PATH'] = training_data_path
    else:
        print("⚠️ No training data channel found, using local data")
        os.environ['DATASET_PATH'] = args.data_dir
    
    # Train model with hyperparameters
    success = train_yolo_with_hyperparameters(args.hyperparameters)
    
    if success:
        # Find best model
        best_model_path = Path("runs/segment/sagemaker_yolo_training/weights/best.pt")
        if best_model_path.exists():
            # Evaluate model
            metrics = evaluate_model(str(best_model_path))
            
            # Save model to SageMaker model directory
            import shutil
            shutil.copy2(best_model_path, Path(args.model_dir) / "model.pt")
            
            # Save metrics
            with open(Path(args.model_dir) / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"✅ Model saved to: {args.model_dir}")
            print(f"✅ Metrics saved: {metrics}")
        else:
            print("❌ Best model not found!")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()
