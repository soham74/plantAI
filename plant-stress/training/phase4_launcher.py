#!/usr/bin/env python3
"""
Phase 4: AWS SageMaker Hyperparameter Tuning Launcher
Complete pipeline for S3 dataset management, SageMaker training, and AMT
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_s3_dataset_upload(data_dir, bucket_name, region="us-east-1"):
    """Upload dataset to S3"""
    print("📤 Step 1: Uploading dataset to S3")
    print("-" * 40)
    
    cmd = [
        sys.executable, "training/phase4_s3_dataset_manager.py",
        "--data-dir", data_dir,
        "--bucket-name", bucket_name,
        "--region", region,
        "--action", "upload"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Dataset upload completed successfully!")
        
        # Extract S3 location from output
        output_lines = result.stdout.split('\n')
        s3_location = None
        for line in output_lines:
            if 'S3 Location:' in line:
                s3_location = line.split('S3 Location:')[1].strip()
                break
        
        return s3_location
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset upload failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def create_sagemaker_container():
    """Create SageMaker training container"""
    print("\n🐳 Step 2: Creating SageMaker Training Container")
    print("-" * 40)
    
    # Create Dockerfile for training container
    dockerfile_content = """
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir \
    ultralytics \
    timm \
    tensorboard \
    scikit-learn \
    matplotlib \
    seaborn \
    boto3 \
    sagemaker-training

# Set working directory
WORKDIR /opt/ml/code

# Copy training scripts
COPY training/phase4_sagemaker_yolo.py .
COPY training/phase4_sagemaker_mobilevit.py .
COPY labels/plantdoc_seg.yaml .

# Set environment variables
ENV PYTHONPATH=/opt/ml/code
ENV SAGEMAKER_PROGRAM=phase4_sagemaker_yolo.py

# Default command
CMD ["python", "phase4_sagemaker_yolo.py"]
"""
    
    dockerfile_path = Path("Dockerfile.sagemaker")
    with open(dockerfile_path, 'w') as f:
        f.write(dockerfile_content)
    
    print(f"✅ Dockerfile created: {dockerfile_path}")
    print("📋 Next steps:")
    print("1. Build container: docker build -f Dockerfile.sagemaker -t plant-stress-training .")
    print("2. Tag for ECR: docker tag plant-stress-training <account>.dkr.ecr.<region>.amazonaws.com/plant-stress-training:latest")
    print("3. Push to ECR: docker push <account>.dkr.ecr.<region>.amazonaws.com/plant-stress-training:latest")
    
    return dockerfile_path

def create_hyperparameter_tuning_job(
    model_type, 
    role_arn, 
    image_uri, 
    s3_data_uri, 
    s3_output_uri,
    region="us-east-1",
    max_jobs=20,
    max_parallel_jobs=4,
    strategy="Bayesian"
):
    """Create hyperparameter tuning job"""
    print(f"\n🎯 Step 3: Creating Hyperparameter Tuning Job ({model_type})")
    print("-" * 40)
    
    cmd = [
        sys.executable, "training/phase4_automatic_model_tuning.py",
        "--model-type", model_type,
        "--role-arn", role_arn,
        "--image-uri", image_uri,
        "--s3-data-uri", s3_data_uri,
        "--s3-output-uri", s3_output_uri,
        "--region", region,
        "--max-jobs", str(max_jobs),
        "--max-parallel-jobs", str(max_parallel_jobs),
        "--strategy", strategy,
        "--action", "create"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Hyperparameter tuning job created successfully!")
        
        # Extract tuning job name from output
        output_lines = result.stdout.split('\n')
        tuning_job_name = None
        for line in output_lines:
            if 'Tuning job name:' in line:
                tuning_job_name = line.split('Tuning job name:')[1].strip()
                break
        
        return tuning_job_name
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Hyperparameter tuning job creation failed: {e}")
        print(f"Error output: {e.stderr}")
        return None

def monitor_tuning_job(tuning_job_name, region="us-east-1"):
    """Monitor hyperparameter tuning job"""
    print(f"\n🔍 Step 4: Monitoring Tuning Job")
    print("-" * 40)
    
    cmd = [
        sys.executable, "training/phase4_automatic_model_tuning.py",
        "--action", "monitor",
        "--tuning-job-name", tuning_job_name,
        "--region", region
    ]
    
    print(f"Command: {' '.join(cmd)}")
    print("Starting monitoring... (Press Ctrl+C to stop)")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Monitoring failed: {e}")

def get_best_results(tuning_job_name, region="us-east-1"):
    """Get best training job results"""
    print(f"\n🏆 Step 5: Getting Best Results")
    print("-" * 40)
    
    cmd = [
        sys.executable, "training/phase4_automatic_model_tuning.py",
        "--action", "get-best",
        "--tuning-job-name", tuning_job_name,
        "--region", region
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Best results retrieved successfully!")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to get best results: {e}")

def create_phase4_summary(
    s3_location, 
    tuning_job_names, 
    bucket_name, 
    region, 
    max_jobs, 
    max_parallel_jobs
):
    """Create Phase 4 summary"""
    summary = {
        "phase": "Phase 4 - Hyperparameter Tuning (AWS SageMaker)",
        "timestamp": datetime.now().isoformat(),
        "s3_dataset_location": s3_location,
        "s3_bucket": bucket_name,
        "aws_region": region,
        "tuning_jobs": tuning_job_names,
        "configuration": {
            "max_jobs": max_jobs,
            "max_parallel_jobs": max_parallel_jobs,
            "strategy": "Bayesian"
        },
        "models_tuned": ["YOLO", "MobileViT"],
        "objective_metrics": {
            "yolo": "mAP50-95 (Maximize)",
            "mobilevit_classification": "macro_f1 (Maximize)",
            "mobilevit_regression": "rmse (Minimize)"
        },
        "hyperparameter_ranges": {
            "yolo": [
                "model_size: ['n', 's', 'm']",
                "img_size: [512, 640]",
                "epochs: [50, 200]",
                "batch_size: [8, 16, 32]",
                "lr0: [0.001, 0.1]",
                "weight_decay: [0.0001, 0.001]",
                "augmentation: HSV, mosaic, flips, blur"
            ],
            "mobilevit": [
                "model_name: [xxs, xs, s]",
                "img_size: [224, 256]",
                "batch_size: [16, 32, 64]",
                "epochs: [50, 150]",
                "lr: [1e-5, 1e-3]",
                "weight_decay: [1e-5, 1e-3]",
                "task: [classification, regression]",
                "loss: [ce, focal, smooth]"
            ]
        }
    }
    
    summary_file = Path("PHASE4_SUMMARY.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📋 Phase 4 summary saved to: {summary_file}")
    return summary

def main():
    parser = argparse.ArgumentParser(description="Phase 4: AWS SageMaker Hyperparameter Tuning Launcher")
    parser.add_argument("--data-dir", default="data_proc", help="Local dataset directory")
    parser.add_argument("--bucket-name", required=True, help="S3 bucket name")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--image-uri", required=True, help="Training container image URI")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--max-jobs", type=int, default=20, help="Maximum number of training jobs")
    parser.add_argument("--max-parallel-jobs", type=int, default=4, help="Maximum parallel jobs")
    parser.add_argument("--strategy", choices=["Bayesian", "Random"], default="Bayesian", help="Tuning strategy")
    parser.add_argument("--skip-upload", action="store_true", help="Skip dataset upload")
    parser.add_argument("--skip-container", action="store_true", help="Skip container creation")
    parser.add_argument("--models", nargs="+", choices=["yolo", "mobilevit"], default=["yolo", "mobilevit"], 
                       help="Models to tune")
    parser.add_argument("--action", choices=["full", "upload", "tune", "monitor"], default="full",
                       help="Action to perform")
    parser.add_argument("--tuning-job-name", help="Tuning job name for monitoring")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 4")
    print("AWS SageMaker Hyperparameter Tuning")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"S3 bucket: {args.bucket_name}")
    print(f"Region: {args.region}")
    print(f"Models: {args.models}")
    print(f"Strategy: {args.strategy}")
    print(f"Max jobs: {args.max_jobs}")
    print(f"Max parallel jobs: {args.max_parallel_jobs}")
    print()
    
    s3_location = None
    tuning_job_names = []
    
    if args.action in ["full", "upload"] and not args.skip_upload:
        # Step 1: Upload dataset to S3
        s3_location = run_s3_dataset_upload(args.data_dir, args.bucket_name, args.region)
        if not s3_location:
            print("❌ Dataset upload failed. Exiting.")
            return
    
    if args.action in ["full"] and not args.skip_container:
        # Step 2: Create SageMaker container
        create_sagemaker_container()
    
    if args.action in ["full", "tune"]:
        # Step 3: Create hyperparameter tuning jobs
        s3_data_uri = s3_location or f"s3://{args.bucket_name}/datasets/"
        s3_output_uri = f"s3://{args.bucket_name}/models/"
        
        for model_type in args.models:
            tuning_job_name = create_hyperparameter_tuning_job(
                model_type,
                args.role_arn,
                args.image_uri,
                s3_data_uri,
                s3_output_uri,
                args.region,
                args.max_jobs,
                args.max_parallel_jobs,
                args.strategy
            )
            
            if tuning_job_name:
                tuning_job_names.append(tuning_job_name)
                print(f"✅ Tuning job created for {model_type}: {tuning_job_name}")
            else:
                print(f"❌ Failed to create tuning job for {model_type}")
    
    if args.action == "monitor" and args.tuning_job_name:
        # Step 4: Monitor tuning job
        monitor_tuning_job(args.tuning_job_name, args.region)
        
        # Step 5: Get best results
        get_best_results(args.tuning_job_name, args.region)
    
    # Create summary
    if args.action in ["full", "tune"] and tuning_job_names:
        summary = create_phase4_summary(
            s3_location,
            tuning_job_names,
            args.bucket_name,
            args.region,
            args.max_jobs,
            args.max_parallel_jobs
        )
        
        print(f"\n🎉 Phase 4 completed successfully!")
        print(f"Tuning jobs created: {len(tuning_job_names)}")
        print(f"Monitor jobs with:")
        for job_name in tuning_job_names:
            print(f"  python {sys.argv[0]} --action monitor --tuning-job-name {job_name}")

if __name__ == "__main__":
    main()
