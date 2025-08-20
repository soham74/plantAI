#!/usr/bin/env python3
"""
Analyze and prepare datasets for plant stress detection
- Count images per class
- Check image formats and sizes
- Create train/val/test splits
- Generate YOLO format labels
"""

import os
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict
import pandas as pd
from PIL import Image
import numpy as np

def analyze_plantvillage():
    """Analyze PlantVillage dataset structure"""
    print("Analyzing PlantVillage dataset...")
    
    plantvillage_path = Path("data_raw/plantvillage")
    if not plantvillage_path.exists():
        print("PlantVillage dataset not found!")
        return None
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(plantvillage_path.rglob(f"*{ext}"))
        image_files.extend(plantvillage_path.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images in PlantVillage")
    
    # Analyze class distribution
    class_counts = defaultdict(int)
    class_samples = defaultdict(list)
    
    for img_path in image_files:
        # Extract class from path (assuming structure: class_name/image.jpg)
        class_name = img_path.parent.name
        class_counts[class_name] += 1
        class_samples[class_name].append(str(img_path))
    
    print(f"Found {len(class_counts)} classes:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} images")
    
    return {
        'total_images': len(image_files),
        'num_classes': len(class_counts),
        'class_counts': dict(class_counts),
        'class_samples': dict(class_samples)
    }

def analyze_plantdoc():
    """Analyze PlantDoc dataset structure"""
    print("Analyzing PlantDoc dataset...")
    
    plantdoc_path = Path("data_raw/plantdoc")
    if not plantdoc_path.exists():
        print("PlantDoc dataset not found!")
        return None
    
    # Analyze train and test splits
    train_path = plantdoc_path / "train"
    test_path = plantdoc_path / "test"
    
    train_stats = analyze_split(train_path) if train_path.exists() else None
    test_stats = analyze_split(test_path) if test_path.exists() else None
    
    return {
        'train': train_stats,
        'test': test_stats
    }

def analyze_split(split_path):
    """Analyze a train/test split"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(split_path.rglob(f"*{ext}"))
        image_files.extend(split_path.rglob(f"*{ext.upper()}"))
    
    class_counts = defaultdict(int)
    class_samples = defaultdict(list)
    
    for img_path in image_files:
        class_name = img_path.parent.name
        class_counts[class_name] += 1
        class_samples[class_name].append(str(img_path))
    
    return {
        'total_images': len(image_files),
        'num_classes': len(class_counts),
        'class_counts': dict(class_counts),
        'class_samples': dict(class_samples)
    }

def analyze_fgvc2020():
    """Analyze FGVC Plant Pathology 2020 dataset"""
    print("Analyzing FGVC Plant Pathology 2020 dataset...")
    
    fgvc_path = Path("data_raw/fgvc2020")
    if not fgvc_path.exists():
        print("FGVC 2020 dataset not found!")
        return None
    
    # Check for train.csv and test.csv
    train_csv = fgvc_path / "train.csv"
    test_csv = fgvc_path / "test.csv"
    
    if train_csv.exists():
        train_df = pd.read_csv(train_csv)
        print(f"Train set: {len(train_df)} images")
        
        # FGVC uses one-hot encoding, so we need to convert to labels
        label_columns = ['healthy', 'multiple_diseases', 'rust', 'scab']
        train_df['labels'] = train_df[label_columns].idxmax(axis=1)
        
        print("Class distribution:")
        print(train_df['labels'].value_counts())
        
        return {
            'train_csv': str(train_csv),
            'test_csv': str(test_csv) if test_csv.exists() else None,
            'train_df': train_df,
            'num_classes': len(label_columns),
            'class_counts': train_df['labels'].value_counts().to_dict()
        }
    
    return None

def create_yolo_labels_plantdoc():
    """Create YOLO format labels for PlantDoc dataset"""
    print("Creating YOLO labels for PlantDoc...")
    
    plantdoc_path = Path("data_raw/plantdoc")
    if not plantdoc_path.exists():
        print("PlantDoc dataset not found!")
        return
    
    # Create labels directory
    labels_dir = Path("labels/plantdoc")
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Create class mapping
    train_path = plantdoc_path / "train"
    classes = sorted([d.name for d in train_path.iterdir() if d.is_dir()])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Save class mapping
    with open(labels_dir / "classes.txt", "w") as f:
        for cls in classes:
            f.write(f"{cls}\n")
    
    print(f"Created class mapping with {len(classes)} classes")
    
    # For PlantDoc, we'll create bounding box labels
    # Since we don't have actual bounding boxes, we'll create full-image boxes
    # This is a placeholder - in practice you'd want to label actual lesions
    
    train_labels_dir = labels_dir / "train"
    train_labels_dir.mkdir(exist_ok=True)
    
    train_path = plantdoc_path / "train"
    for class_dir in train_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_idx = class_to_idx[class_dir.name]
        
        for img_path in class_dir.glob("*.jpg"):
            # Create YOLO label file
            label_path = train_labels_dir / f"{img_path.stem}.txt"
            
            # Full image bounding box (normalized coordinates)
            # Format: class_id center_x center_y width height
            with open(label_path, "w") as f:
                f.write(f"{class_idx} 0.5 0.5 1.0 1.0\n")
    
    print(f"Created YOLO labels for {len(list(train_labels_dir.glob('*.txt')))} images")

def create_train_val_test_splits():
    """Create train/val/test splits for the datasets"""
    print("Creating train/val/test splits...")
    
    # Create processed data directory
    proc_dir = Path("data_proc")
    proc_dir.mkdir(exist_ok=True)
    
    # For now, we'll focus on PlantDoc as it's more suitable for field conditions
    plantdoc_path = Path("data_raw/plantdoc")
    if not plantdoc_path.exists():
        print("PlantDoc dataset not found!")
        return
    
    # Create splits
    splits = {
        'train': 0.7,
        'val': 0.2,
        'test': 0.1
    }
    
    train_path = plantdoc_path / "train"
    for class_dir in train_path.iterdir():
        if not class_dir.is_dir():
            continue
            
        class_name = class_dir.name
        images = list(class_dir.glob("*.jpg"))
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * splits['train'])
        n_val = int(n_total * splits['val'])
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create symbolic links to processed directory (to save disk space)
        for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            split_dir = proc_dir / split_name / class_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in split_images:
                link_path = split_dir / img_path.name
                if not link_path.exists():
                    link_path.symlink_to(img_path)
        
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

def main():
    """Main analysis function"""
    print("Starting dataset analysis...")
    
    # Analyze datasets
    plantvillage_stats = analyze_plantvillage()
    plantdoc_stats = analyze_plantdoc()
    fgvc_stats = analyze_fgvc2020()
    
    # Save analysis results
    analysis_results = {
        'plantvillage': plantvillage_stats,
        'plantdoc': plantdoc_stats,
        'fgvc2020': fgvc_stats
    }
    
    with open("data_raw/dataset_analysis.json", "w") as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print("\nAnalysis saved to data_raw/dataset_analysis.json")
    
    # Create YOLO labels
    create_yolo_labels_plantdoc()
    
    # Create train/val/test splits
    create_train_val_test_splits()
    
    print("\nDataset preparation complete!")

if __name__ == "__main__":
    main()
