#!/usr/bin/env python3
"""
Phase 4: S3 Dataset Manager
Upload datasets to S3 for SageMaker training and hyperparameter tuning
"""

import os
import sys
import argparse
import json
import boto3
import zipfile
from pathlib import Path
from datetime import datetime
import hashlib
from botocore.exceptions import ClientError

def create_s3_client(region_name="us-east-1"):
    """Create S3 client with error handling"""
    try:
        s3_client = boto3.client('s3', region_name=region_name)
        # Test connection
        s3_client.list_buckets()
        print(f"✅ S3 client created successfully for region: {region_name}")
        return s3_client
    except ClientError as e:
        print(f"❌ Failed to create S3 client: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error creating S3 client: {e}")
        return None

def create_s3_bucket(s3_client, bucket_name, region_name="us-east-1"):
    """Create S3 bucket if it doesn't exist"""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' already exists")
        return True
    except ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            # Bucket doesn't exist, create it
            try:
                if region_name == 'us-east-1':
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(
                        Bucket=bucket_name,
                        CreateBucketConfiguration={'LocationConstraint': region_name}
                    )
                print(f"✅ Created bucket '{bucket_name}' in region '{region_name}'")
                return True
            except ClientError as create_error:
                print(f"❌ Failed to create bucket '{bucket_name}': {create_error}")
                return False
        else:
            print(f"❌ Error checking bucket '{bucket_name}': {e}")
            return False

def calculate_dataset_hash(data_dir):
    """Calculate hash of dataset for versioning"""
    hash_md5 = hashlib.md5()
    
    for root, dirs, files in os.walk(data_dir):
        for file in sorted(files):
            if file.endswith(('.jpg', '.jpeg', '.png', '.txt', '.json', '.yaml')):
                file_path = os.path.join(root, file)
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
    
    return hash_md5.hexdigest()

def create_dataset_archive(data_dir, output_path):
    """Create zip archive of dataset"""
    print(f"📦 Creating dataset archive: {output_path}")
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png', '.txt', '.json', '.yaml')):
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, data_dir)
                    zipf.write(file_path, arcname)
                    print(f"  Added: {arcname}")
    
    print(f"✅ Dataset archive created: {output_path}")
    return output_path

def upload_to_s3(s3_client, local_path, bucket_name, s3_key):
    """Upload file to S3 with progress tracking"""
    try:
        file_size = os.path.getsize(local_path)
        print(f"📤 Uploading {local_path} to s3://{bucket_name}/{s3_key}")
        print(f"   File size: {file_size / (1024*1024):.2f} MB")
        
        # Upload with progress callback
        def progress_callback(bytes_transferred):
            if file_size > 0:
                percent = (bytes_transferred / file_size) * 100
                print(f"\r   Progress: {percent:.1f}%", end='', flush=True)
        
        s3_client.upload_file(
            local_path, 
            bucket_name, 
            s3_key,
            Callback=progress_callback
        )
        print(f"\n✅ Upload completed: s3://{bucket_name}/{s3_key}")
        return True
        
    except ClientError as e:
        print(f"\n❌ Upload failed: {e}")
        return False

def upload_dataset_to_s3(data_dir, bucket_name, s3_prefix="datasets", region_name="us-east-1"):
    """Upload dataset to S3 with versioning"""
    
    # Create S3 client
    s3_client = create_s3_client(region_name)
    if not s3_client:
        return False
    
    # Create bucket if needed
    if not create_s3_bucket(s3_client, bucket_name, region_name):
        return False
    
    # Calculate dataset hash for versioning
    dataset_hash = calculate_dataset_hash(data_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create archive
    archive_name = f"plant_stress_dataset_{timestamp}_{dataset_hash[:8]}.zip"
    archive_path = Path("temp") / archive_name
    archive_path.parent.mkdir(exist_ok=True)
    
    if not create_dataset_archive(data_dir, archive_path):
        return False
    
    # Upload to S3
    s3_key = f"{s3_prefix}/{archive_name}"
    success = upload_to_s3(s3_client, archive_path, bucket_name, s3_key)
    
    # Clean up local archive
    if archive_path.exists():
        archive_path.unlink()
    
    if success:
        # Create dataset manifest
        manifest = {
            "dataset_name": "plant_stress_dataset",
            "timestamp": timestamp,
            "hash": dataset_hash,
            "s3_location": f"s3://{bucket_name}/{s3_key}",
            "size_mb": os.path.getsize(archive_path) / (1024*1024) if archive_path.exists() else 0,
            "classes": len([d for d in Path(data_dir).iterdir() if d.is_dir()]),
            "total_images": sum(len(list(Path(data_dir).glob(f"**/*.{ext}"))) for ext in ['jpg', 'jpeg', 'png'])
        }
        
        manifest_key = f"{s3_prefix}/manifest_{timestamp}.json"
        with open("temp_manifest.json", 'w') as f:
            json.dump(manifest, f, indent=2)
        
        upload_to_s3(s3_client, "temp_manifest.json", bucket_name, manifest_key)
        os.remove("temp_manifest.json")
        
        print(f"✅ Dataset uploaded successfully!")
        print(f"   S3 Location: s3://{bucket_name}/{s3_key}")
        print(f"   Manifest: s3://{bucket_name}/{manifest_key}")
        
        return {
            "s3_location": f"s3://{bucket_name}/{s3_key}",
            "manifest": manifest
        }
    
    return False

def list_datasets_in_s3(s3_client, bucket_name, s3_prefix="datasets"):
    """List available datasets in S3"""
    try:
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=s3_prefix
        )
        
        if 'Contents' in response:
            print(f"📋 Available datasets in s3://{bucket_name}/{s3_prefix}:")
            for obj in response['Contents']:
                if obj['Key'].endswith('.zip'):
                    print(f"  📦 {obj['Key']}")
                    print(f"     Size: {obj['Size'] / (1024*1024):.2f} MB")
                    print(f"     Modified: {obj['LastModified']}")
                elif obj['Key'].endswith('manifest.json'):
                    print(f"  📄 {obj['Key']}")
        else:
            print(f"No datasets found in s3://{bucket_name}/{s3_prefix}")
            
    except ClientError as e:
        print(f"❌ Error listing datasets: {e}")

def main():
    parser = argparse.ArgumentParser(description="Phase 4: S3 Dataset Manager")
    parser.add_argument("--data-dir", default="data_proc", help="Local dataset directory")
    parser.add_argument("--bucket-name", required=True, help="S3 bucket name")
    parser.add_argument("--s3-prefix", default="datasets", help="S3 prefix for datasets")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--action", choices=["upload", "list"], default="upload", 
                       help="Action to perform")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 4")
    print("S3 Dataset Manager")
    print("=" * 60)
    print(f"Action: {args.action}")
    print(f"Bucket: {args.bucket_name}")
    print(f"Region: {args.region}")
    print()
    
    if args.action == "upload":
        if not Path(args.data_dir).exists():
            print(f"❌ Data directory not found: {args.data_dir}")
            return
        
        result = upload_dataset_to_s3(
            args.data_dir, 
            args.bucket_name, 
            args.s3_prefix, 
            args.region
        )
        
        if result:
            print("\n🎉 Dataset upload completed successfully!")
            print("Ready for SageMaker training and hyperparameter tuning.")
        else:
            print("\n❌ Dataset upload failed!")
    
    elif args.action == "list":
        s3_client = create_s3_client(args.region)
        if s3_client:
            list_datasets_in_s3(s3_client, args.bucket_name, args.s3_prefix)

if __name__ == "__main__":
    main()
