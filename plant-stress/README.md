# Plant Stress Detection System

A comprehensive AI system for detecting plant stress and diseases using computer vision, combining CNNs and Transformers for real-time field analysis.

## Overview

This project implements a hybrid CNN + Transformer approach for plant stress detection, designed to work with iPhone camera feeds on laptops for real-time field analysis. The system includes:

- **Stage A**: YOLO-based lesion/leaf detection and segmentation
- **Stage B**: MobileViT classifier for stress classification and severity regression
- **Geospatial Integration**: GPS fusion and field-level aggregation
- **Real-time Processing**: Live iPhone camera feed processing

## Repository Structure

```
plant-stress/
├── data_raw/                 # Downloaded datasets (PlantVillage, PlantDoc, FGVC)
├── data_proc/                # Cleaned/augmented datasets
├── labels/                   # YOLO + COCO/Label Studio exports
├── training/
│   ├── detect_seg/           # Ultralytics YOLO (seg/det)
│   ├── classify_pt/          # PyTorch timm MobileViT/EfficientNet-Lite
│   ├── classify_tf/          # TensorFlow EfficientNet-Lite/MobileNetV3
│   └── hpo/                  # SageMaker AMT jobspecs, ranges
├── runtime/
│   ├── stream/               # Laptop livestream (OpenCV), GPS fusion, overlay
│   └── onnx/                 # Optional ONNX/ORT acceleration
├── geo/
│   ├── ingest/               # Write predictions as rows with GPS
│   └── aggregate/            # GeoPandas/GDAL to GeoJSON/GeoTIFF
├── eval/
│   ├── metrics/              # PR, mAP, F1, RMSE; confusion matrices
│   └── viz/                  # TensorBoard + panels
└── scripts/                  # One-off utilities (download, split, convert)
```

## Phase 1: Data Acquisition and Preparation

### Datasets

1. **PlantVillage**: Large leaf close-ups for pretraining/baseline
2. **PlantDoc**: Field-like images closer to phone photos
3. **FGVC Plant Pathology 2020**: Apple leaves with real-world noise

### Setup Instructions

1. **Environment Setup**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Download Datasets**:
   ```bash
   # Install Kaggle CLI and configure credentials
   pip install kaggle
   # Configure kaggle.json with your API credentials
   
   # Download datasets
   python scripts/download_datasets.py
   ```

3. **Analyze and Prepare Data**:
   ```bash
   python scripts/analyze_datasets.py
   ```

4. **Preprocess Images** (optional):
   ```bash
   # Basic preprocessing
   python scripts/preprocess_images.py --input data_raw/plantdoc --output data_proc/plantdoc_processed --size 512
   
   # With color constancy
   python scripts/preprocess_images.py --input data_raw/plantdoc --output data_proc/plantdoc_processed --size 512 --color-constancy --method gray_world
   
   # With VARI channel computation
   python scripts/preprocess_images.py --input data_raw/plantdoc --output data_proc/plantdoc_processed --size 512 --vari --save-vari
   ```

## Phase 2: Model Training

### YOLO Detection/Segmentation

```bash
# Train YOLO segmentation model
yolo task=segment mode=train model=yolo11n-seg.pt data=labels/plantdoc_seg.yaml \
     imgsz=512 epochs=100 batch=16 amp=True workers=8

# Validate model
yolo mode=val model=runs/segment/train/weights/best.pt

# Export to ONNX
yolo mode=export model=runs/segment/train/weights/best.pt format=onnx
```

### MobileViT Classifier

```bash
# Train MobileViT classifier
python training/classify_pt/train.py --data data_proc --model mobilevit_xxs --epochs 100
```

## Phase 3: Real-time Processing

### Live iPhone Camera Feed

```bash
# Run live processing pipeline
python runtime/stream/run_live.py --camera 2 --model-dir runs/segment/train/weights/
```

### GPS Integration

The system supports GPS fusion for field-level analysis:

```bash
# Start GPS server on iPhone (manual setup required)
# Run processing with GPS fusion
python runtime/stream/run_live_gps.py --gps-server 192.168.1.100:8080
```

## Phase 4: Geospatial Analysis

```bash
# Aggregate predictions to field-level
python geo/aggregate/grid_agg.py --input predictions.csv --output stress_heatmap.tif
```

## Key Features

### Color Constancy
- **Gray-World**: Assumes average scene color should be gray
- **Shades-of-Gray**: Uses Minkowski norm for robust correction

### VARI Channel
- **Visible Atmospherically Resistant Index**: `VARI = (G - R) / (G + R - B)`
- Useful for vegetation analysis and stress detection

### Model Architecture
- **Detection**: YOLO-n/s for real-time lesion/leaf localization
- **Classification**: MobileViT-xxs/xs/s for stress classification
- **Hybrid Approach**: Combines CNN speed with Transformer accuracy

## Performance Targets

- **Real-time**: ≥10-15 FPS on laptop
- **Detection**: mAP@0.5:0.95 ≥ baseline
- **Classification**: macro-F1 ≥ baseline
- **End-to-end**: GPS-tagged predictions with field aggregation

## Requirements

- **Hardware**: Apple Silicon Mac or NVIDIA GPU laptop
- **Software**: Python 3.8+, PyTorch 2.0+, Ultralytics 8.3+
- **Data**: ~10GB for full datasets
- **Storage**: 50GB+ for processed data and models

## Next Steps

1. **Phase 1**: Complete dataset preparation and analysis
2. **Phase 2**: Train YOLO detection model
3. **Phase 3**: Train MobileViT classifier
4. **Phase 4**: Implement real-time pipeline
5. **Phase 5**: Add GPS fusion and geospatial analysis

## Contributing

This project follows a phased development approach. Each phase builds upon the previous one, ensuring a robust and scalable system for plant stress detection.

## License

MIT License - see LICENSE file for details.
