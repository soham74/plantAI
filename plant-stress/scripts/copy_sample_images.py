#!/usr/bin/env python3
"""
Copy sample images from dataset for testing
"""

import os
import shutil
import random
from pathlib import Path

def copy_sample_images():
    """Copy sample images from dataset to examples"""
    print("📸 Copying sample images for testing...")
    
    # Source and destination
    src_dir = Path("data/data_proc/test")
    dst_dir = Path("examples/sample_images")
    
    # Create destination directory
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all images
    all_images = []
    for class_dir in src_dir.iterdir():
        if class_dir.is_dir():
            images = list(class_dir.glob("*.jpg"))
            all_images.extend(images)
    
    # Select random samples (up to 5)
    sample_count = min(5, len(all_images))
    sample_images = random.sample(all_images, sample_count)
    
    # Copy images
    for i, img_path in enumerate(sample_images):
        # Create descriptive name
        class_name = img_path.parent.name.replace(" ", "_").lower()
        new_name = f"sample_{i+1}_{class_name}.jpg"
        dst_path = dst_dir / new_name
        
        # Copy file
        shutil.copy2(img_path, dst_path)
        print(f"  📁 Copied: {new_name}")
    
    print(f"✅ Copied {len(sample_images)} sample images to examples/sample_images/")
    print("💡 You can now use these images to test the model!")

if __name__ == "__main__":
    copy_sample_images()
