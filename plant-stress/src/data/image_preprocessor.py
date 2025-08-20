#!/usr/bin/env python3
"""
Image preprocessing for plant stress detection
- Resize images to standard sizes
- Color constancy (Gray-World / Shades-of-Gray)
- VARI channel computation
- Data augmentation
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import argparse
from tqdm import tqdm

def gray_world_color_constancy(image):
    """
    Apply Gray-World color constancy
    Assumes that the average color in a scene should be gray
    """
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # Calculate mean for each channel
    means = np.mean(img_float, axis=(0, 1))
    
    # Calculate scaling factors to make means equal
    gray_mean = np.mean(means)
    scaling_factors = gray_mean / (means + 1e-8)
    
    # Apply scaling
    corrected = img_float * scaling_factors[np.newaxis, np.newaxis, :]
    
    # Clip to valid range and convert back to uint8
    corrected = np.clip(corrected, 0, 1) * 255
    return corrected.astype(np.uint8)

def shades_of_gray_color_constancy(image, p=6):
    """
    Apply Shades-of-Gray color constancy
    Uses Minkowski norm instead of simple mean
    """
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # Calculate Minkowski norm for each channel
    norms = np.power(np.mean(np.power(img_float, p), axis=(0, 1)), 1/p)
    
    # Calculate scaling factors
    gray_norm = np.mean(norms)
    scaling_factors = gray_norm / (norms + 1e-8)
    
    # Apply scaling
    corrected = img_float * scaling_factors[np.newaxis, np.newaxis, :]
    
    # Clip to valid range and convert back to uint8
    corrected = np.clip(corrected, 0, 1) * 255
    return corrected.astype(np.uint8)

def compute_vari_channel(image):
    """
    Compute VARI (Visible Atmospherically Resistant Index) channel
    VARI = (G - R) / (G + R - B)
    """
    # Convert to float
    img_float = image.astype(np.float32) / 255.0
    
    # Extract channels
    r, g, b = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    
    # Compute VARI
    denominator = g + r - b
    vari = np.divide(g - r, denominator + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Normalize to [0, 1] range
    vari = (vari - np.min(vari)) / (np.max(vari) - np.min(vari) + 1e-8)
    
    return (vari * 255).astype(np.uint8)

def resize_image(image, target_size, maintain_aspect=True):
    """
    Resize image to target size
    """
    if maintain_aspect:
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 1:  # Width > height
            new_w = target_size
            new_h = int(target_size / aspect_ratio)
        else:  # Height >= width
            new_h = target_size
            new_w = int(target_size * aspect_ratio)
        
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        return padded
    else:
        return cv2.resize(image, (target_size, target_size))

def preprocess_image(image_path, output_path, target_size=512, 
                    apply_color_constancy=False, color_constancy_method='gray_world',
                    compute_vari=False, save_vari=False):
    """
    Preprocess a single image
    """
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Could not read image: {image_path}")
        return False
    
    # Apply color constancy if requested
    if apply_color_constancy:
        if color_constancy_method == 'gray_world':
            image = gray_world_color_constancy(image)
        elif color_constancy_method == 'shades_of_gray':
            image = shades_of_gray_color_constancy(image)
    
    # Resize image
    resized = resize_image(image, target_size)
    
    # Save processed image
    cv2.imwrite(str(output_path), resized)
    
    # Compute and save VARI channel if requested
    if compute_vari:
        vari_channel = compute_vari_channel(image)
        if save_vari:
            vari_path = output_path.parent / f"{output_path.stem}_vari{output_path.suffix}"
            cv2.imwrite(str(vari_path), vari_channel)
    
    return True

def preprocess_dataset(input_dir, output_dir, target_size=512,
                      apply_color_constancy=False, color_constancy_method='gray_world',
                      compute_vari=False, save_vari=False):
    """
    Preprocess entire dataset
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    if not input_path.exists():
        print(f"Input directory does not exist: {input_dir}")
        return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(input_path.rglob(f"*{ext}"))
        image_files.extend(input_path.rglob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    success_count = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        # Create relative path structure
        rel_path = img_path.relative_to(input_path)
        output_file = output_path / rel_path
        
        # Create output directory
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process image
        if preprocess_image(img_path, output_file, target_size,
                          apply_color_constancy, color_constancy_method,
                          compute_vari, save_vari):
            success_count += 1
    
    print(f"Successfully processed {success_count}/{len(image_files)} images")

def main():
    parser = argparse.ArgumentParser(description="Preprocess images for plant stress detection")
    parser.add_argument("--input", required=True, help="Input directory containing images")
    parser.add_argument("--output", required=True, help="Output directory for processed images")
    parser.add_argument("--size", type=int, default=512, help="Target image size")
    parser.add_argument("--color-constancy", action="store_true", 
                       help="Apply color constancy correction")
    parser.add_argument("--method", choices=['gray_world', 'shades_of_gray'], 
                       default='gray_world', help="Color constancy method")
    parser.add_argument("--vari", action="store_true", 
                       help="Compute VARI channel")
    parser.add_argument("--save-vari", action="store_true", 
                       help="Save VARI channel as separate image")
    
    args = parser.parse_args()
    
    preprocess_dataset(args.input, args.output, args.size,
                      args.color_constancy, args.method,
                      args.vari, args.save_vari)

if __name__ == "__main__":
    main()
