#!/usr/bin/env python3
"""
Prepare training data for plant stress detection
Copy PlantDoc dataset to data_proc with proper structure
"""

import os
import shutil
from pathlib import Path

def prepare_plantdoc_data():
    """Prepare PlantDoc dataset for training"""
    print("🌱 Preparing PlantDoc dataset for training...")
    
    # Source and destination paths
    src_train = Path("../../Downloads/PlantDoc-Dataset/train")
    src_test = Path("../../Downloads/PlantDoc-Dataset/test")
    dst_train = Path("data_proc/train")
    dst_val = Path("data_proc/val")
    dst_test = Path("data_proc/test")
    
    # Create directories
    dst_train.mkdir(parents=True, exist_ok=True)
    dst_val.mkdir(parents=True, exist_ok=True)
    dst_test.mkdir(parents=True, exist_ok=True)
    
    # Copy test data
    print("📁 Copying test data...")
    if src_test.exists():
        for class_dir in src_test.iterdir():
            if class_dir.is_dir():
                dst_class = dst_test / class_dir.name
                dst_class.mkdir(exist_ok=True)
                for img_file in class_dir.glob("*.jpg"):
                    shutil.copy2(img_file, dst_class)
    
    # Copy train data (split into train/val)
    print("📁 Copying and splitting train data...")
    if src_train.exists():
        for class_dir in src_train.iterdir():
            if class_dir.is_dir():
                # Create class directories
                dst_train_class = dst_train / class_dir.name
                dst_val_class = dst_val / class_dir.name
                dst_train_class.mkdir(exist_ok=True)
                dst_val_class.mkdir(exist_ok=True)
                
                # Get all images for this class
                images = list(class_dir.glob("*.jpg"))
                
                # Split 80% train, 20% val
                split_idx = int(len(images) * 0.8)
                train_images = images[:split_idx]
                val_images = images[split_idx:]
                
                # Copy train images
                for img_file in train_images:
                    shutil.copy2(img_file, dst_train_class)
                
                # Copy val images
                for img_file in val_images:
                    shutil.copy2(img_file, dst_val_class)
                
                print(f"  {class_dir.name}: {len(train_images)} train, {len(val_images)} val")
    
    print("✅ Data preparation completed!")
    
    # Count images
    train_count = sum(len(list(d.glob("*.jpg"))) for d in dst_train.iterdir() if d.is_dir())
    val_count = sum(len(list(d.glob("*.jpg"))) for d in dst_val.iterdir() if d.is_dir())
    test_count = sum(len(list(d.glob("*.jpg"))) for d in dst_test.iterdir() if d.is_dir())
    
    print(f"📊 Dataset summary:")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Test: {test_count} images")
    print(f"  Total: {train_count + val_count + test_count} images")

if __name__ == "__main__":
    prepare_plantdoc_data()
