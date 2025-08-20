#!/usr/bin/env python3
"""
Download datasets for plant stress detection
- PlantVillage: Large leaf close-ups for pretraining/baseline
- PlantDoc: Field-like images closer to phone photos
- FGVC Plant Pathology 2020: Apple leaves with real-world noise
"""

import os
import sys
import subprocess
import zipfile
import requests
from pathlib import Path

def check_kaggle_installed():
    """Check if Kaggle CLI is installed"""
    try:
        subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def download_plantvillage():
    """Download PlantVillage dataset from Kaggle"""
    print("Downloading PlantVillage dataset...")
    
    if not check_kaggle_installed():
        print("Kaggle CLI not found. Please install it first:")
        print("pip install kaggle")
        print("Then configure your Kaggle API credentials")
        return False
    
    try:
        # Create directory
        os.makedirs("data_raw/plantvillage", exist_ok=True)
        
        # Download from Kaggle
        cmd = [
            "kaggle", "datasets", "download", 
            "-d", "abdallahalidev/plantvillage-dataset",
            "-p", "data_raw"
        ]
        subprocess.run(cmd, check=True)
        
        # Extract
        zip_files = list(Path("data_raw").glob("*.zip"))
        for zip_file in zip_files:
            if "plantvillage" in zip_file.name.lower():
                print(f"Extracting {zip_file}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall("data_raw/plantvillage")
                os.remove(zip_file)
                break
        
        print("PlantVillage dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading PlantVillage: {e}")
        return False

def download_plantdoc():
    """Download PlantDoc dataset from GitHub"""
    print("Downloading PlantDoc dataset...")
    
    try:
        # Clone from GitHub
        cmd = [
            "git", "clone", 
            "https://github.com/pratikkayal/PlantDoc-Dataset",
            "data_raw/plantdoc"
        ]
        subprocess.run(cmd, check=True)
        
        print("PlantDoc dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading PlantDoc: {e}")
        return False

def download_fgvc2020():
    """Download FGVC Plant Pathology 2020 dataset from Kaggle"""
    print("Downloading FGVC Plant Pathology 2020 dataset...")
    
    if not check_kaggle_installed():
        print("Kaggle CLI not found. Please install it first:")
        print("pip install kaggle")
        print("Then configure your Kaggle API credentials")
        return False
    
    try:
        # Create directory
        os.makedirs("data_raw/fgvc2020", exist_ok=True)
        
        # Download from Kaggle
        cmd = [
            "kaggle", "competitions", "download",
            "-c", "plant-pathology-2020-fgvc7",
            "-p", "data_raw"
        ]
        subprocess.run(cmd, check=True)
        
        # Extract
        zip_files = list(Path("data_raw").glob("*.zip"))
        for zip_file in zip_files:
            if "fgvc7" in zip_file.name.lower():
                print(f"Extracting {zip_file}...")
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    zip_ref.extractall("data_raw/fgvc2020")
                os.remove(zip_file)
                break
        
        print("FGVC Plant Pathology 2020 dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading FGVC 2020: {e}")
        return False

def main():
    """Main download function"""
    print("Starting dataset downloads...")
    
    # Create data_raw directory
    os.makedirs("data_raw", exist_ok=True)
    
    # Download datasets
    success_count = 0
    
    if download_plantvillage():
        success_count += 1
    
    if download_plantdoc():
        success_count += 1
    
    if download_fgvc2020():
        success_count += 1
    
    print(f"\nDownload complete! {success_count}/3 datasets downloaded successfully.")
    
    if success_count < 3:
        print("\nSome downloads failed. Please check:")
        print("1. Kaggle CLI installation and credentials")
        print("2. Internet connection")
        print("3. Available disk space")

if __name__ == "__main__":
    main()
