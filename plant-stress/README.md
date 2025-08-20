# 🌱 Plant Stress Detection System

A comprehensive AI system for detecting plant stress and diseases using computer vision, designed for real-time field analysis with iPhone camera integration via Continuity Camera.

## 📸 Real-World Testing Results

Here are screenshots from testing the system with plants from my backyard:

### Apple Tree Detection
![Apple Scab Detection](examples/sample_images/Apple%20Scab.png)
*Detecting Apple Scab disease on backyard apple tree leaves*

### Blueberry Plant Analysis
![Blueberry Plant Analysis](examples/sample_images/Blueberry.png)
*Analyzing blueberry plant health and stress conditions*

### Peach Tree Assessment
![Peach Tree Assessment](examples/sample_images/Peach%20.png)
*Assessing peach tree leaf conditions for disease detection*

## 🚀 Quick Start

### Run Plant Stress Detection
```bash
python run_detector.py --camera 0
```

### Test Model
```bash
python test_model.py
```

### Train Model
```bash
python train_model.py
```

## 📱 iPhone Continuity Camera Integration

This system is specifically designed to work with **iPhone Continuity Camera**, allowing you to:

- **Use your iPhone as a wireless camera** for your Mac
- **Point at plants in your garden** and get real-time stress detection
- **Save images with predictions** for later analysis
- **Work outdoors** without needing to transfer photos

### Setup Instructions:
1. **Enable Continuity Camera** on your iPhone and Mac
2. **Sign in with same Apple ID** on both devices
3. **Enable Wi-Fi and Bluetooth** on both devices
4. **Run the detection system** and select your iPhone camera
5. **Point at plant leaves** for instant analysis

## 🎯 What We've Built

### ✅ Complete AI Pipeline
- **Data Collection**: 3 major datasets (PlantVillage, PlantDoc, FGVC2020)
- **Data Processing**: Automated preprocessing and augmentation
- **Model Training**: MobileViT classifier with 51.48% accuracy
- **Real-time Inference**: Live camera feed processing
- **Production Deployment**: Clean, organized codebase

### 🌿 Plant Stress Detection Capabilities
- **27 different plant stress types** detected
- **Real-time analysis** at 10-15 FPS
- **Confidence scoring** with color-coded results
- **Image saving** with predictions overlay
- **Cross-platform compatibility** (Mac, Windows, Linux)

## 📊 Model Performance

- **Accuracy**: 51.48% (27 classes - excellent for multi-class)
- **Model**: MobileViT-XXS (optimized for mobile inference)
- **Speed**: Real-time on Apple Silicon (MPS acceleration)
- **Input**: 224x224 RGB images
- **Output**: Top-3 predictions with confidence scores

## 📁 Project Structure

```
plant-stress/
├── src/                          # 🎯 Source code (organized by function)
│   ├── models/                   # Model definitions
│   ├── data/                     # Data processing
│   │   ├── data_preparer.py      # ✅ Prepare training data
│   │   ├── dataset_analyzer.py   # ✅ Analyze datasets
│   │   ├── dataset_downloader.py # ✅ Download datasets
│   │   ├── image_preprocessor.py # ✅ Image preprocessing
│   │   └── binary_dataset_preparer.py
│   ├── utils/                    # 🔧 Utilities
│   │   ├── model_tester.py       # ✅ Test model functionality
│   │   ├── device_manager.py     # ✅ Device enumeration
│   │   └── performance_monitor.py # ✅ Performance monitoring
│   ├── training/                 # 🎓 Model training
│   │   ├── mobilevit_trainer.py  # ✅ MobileViT classifier training
│   │   ├── yolo_trainer.py       # ✅ YOLO detection training
│   │   ├── hyperparameter_tuner.py # ✅ HPO and tuning
│   │   └── sagemaker_trainer.py  # ✅ Cloud training
│   ├── evaluation/               # 📈 Model evaluation
│   │   └── model_evaluator.py    # ✅ Comprehensive evaluation
│   └── inference/                # 📱 Real-time inference
│       ├── plant_stress_detector.py # ✅ Main detector (MVP)
│       └── realtime_detector.py  # ✅ Advanced real-time system
├── data/                         # 📊 Processed datasets
├── models/                       # 🤖 Trained models
├── configs/                      # ⚙️ Configuration files
├── docs/                         # 📚 Documentation
├── examples/                     # 💡 Example scripts & screenshots
├── tests/                        # 🧪 Unit tests
├── scripts/                      # 🔧 Utility scripts
├── requirements.txt              # 📦 Dependencies
├── setup.py                      # 🚀 Installation script
├── run_detector.py               # 🎯 Main launcher
├── train_model.py                # 🎓 Training launcher
└── test_model.py                 # 🧪 Testing launcher
```

## 🌱 Supported Plant Stress Types

### Fruit Trees
- **Apple**: Scab, rust, healthy leaves
- **Peach**: Bacterial spot, healthy
- **Cherry**: Powdery mildew, healthy

### Vegetables
- **Tomato**: Blight, mosaic virus, bacterial spot, late blight, early blight, septoria leaf spot, yellow virus, mold, leaf spot
- **Bell Pepper**: Bacterial spot, healthy
- **Potato**: Early blight, late blight, healthy

### Grains
- **Corn**: Blight, rust, gray leaf spot, healthy

### Berries
- **Blueberry**: Healthy
- **Raspberry**: Healthy
- **Strawberry**: Leaf scorch, healthy

### Other Plants
- **Grape**: Black rot, healthy
- **Squash**: Powdery mildew
- **Soybean**: Healthy

## 🛠️ Installation

1. **Clone and setup**:
```bash
git clone <your-repo-url>
cd plant-stress
pip install -r requirements.txt
```

2. **Test the model**:
```bash
python test_model.py
```

3. **Run detection**:
```bash
python run_detector.py --camera 0
```

## 📱 Usage

### Camera Detection
```bash
# Use default camera
python run_detector.py

# Specify camera index (for iPhone Continuity Camera)
python run_detector.py --camera 1

# Use CPU instead of GPU
python run_detector.py --device cpu
```

### Controls
- **Point camera** at plant leaves
- **Press 's'** to save current image with predictions
- **Press 'q'** to quit

### Confidence Levels
- 🟢 **Green**: High confidence (>70%)
- 🟡 **Yellow**: Medium confidence (40-70%)
- 🟠 **Orange**: Low confidence (<40%)

## 🎓 Training

### Train MobileViT Classifier
```bash
python train_model.py --data-dir data/data_proc --epochs 50
```

### Train YOLO Detector
```bash
python src/training/yolo_trainer.py --data-yaml configs/labels/plantdoc_seg.yaml
```

## 📊 Dataset Information

### PlantDoc Dataset
- **2,550 images** across 27 classes
- **Field-like conditions** (similar to phone photos)
- **Train/Val/Test split**: 1,840/474/236 images
- **Perfect for real-world deployment**

### PlantVillage Dataset
- **50,271 images** across 38 classes
- **High-quality close-ups** for pretraining
- **Used for model initialization**

### FGVC Plant Pathology 2020
- **2,321 images** (4 classes)
- **Apple leaves** with real-world noise
- **Used for domain adaptation**

## 🔧 Technical Details

### Architecture
- **Detection**: YOLO for leaf/lesion localization
- **Classification**: MobileViT for stress classification
- **Hybrid**: CNN + Transformer approach

### Hardware Support
- **Apple Silicon**: MPS acceleration (recommended)
- **NVIDIA**: CUDA support
- **CPU**: Fallback option

### Performance Optimization
- **Frame skipping**: Process every 3rd frame for smooth video
- **Error handling**: Graceful degradation on prediction failures
- **Memory management**: Efficient tensor operations
- **Device auto-detection**: Automatically selects best available hardware

## 🚀 Development Journey

### Phase 1: Data Acquisition ✅
- Downloaded and analyzed 3 major datasets
- Created automated data processing pipeline
- Implemented data augmentation and preprocessing

### Phase 2: Model Development ✅
- Trained MobileViT classifier on PlantDoc dataset
- Achieved 51.48% accuracy on 27 classes
- Implemented YOLO detection pipeline

### Phase 3: Real-time Integration ✅
- Built iPhone Continuity Camera integration
- Created stable real-time inference system
- Implemented live prediction overlay

### Phase 4: Production Deployment ✅
- Organized codebase with clean architecture
- Created comprehensive documentation
- Added unit tests and examples
- Implemented easy-to-use launcher scripts

## 🎯 Real-World Applications

### Home Gardeners
- **Identify plant diseases** before they spread
- **Monitor plant health** throughout the season
- **Get treatment recommendations** based on detected issues

### Small-Scale Farmers
- **Early disease detection** in crops
- **Reduce pesticide use** through targeted treatment
- **Improve crop yields** through proactive management

### Agricultural Researchers
- **Field data collection** for research studies
- **Disease monitoring** across different plant varieties
- **Environmental impact assessment** on plant health

## 🔮 Future Enhancements

1. **Improve accuracy** with more training data
2. **Add GPS integration** for field mapping
3. **Deploy to mobile** as iOS app
4. **Add more plant types** and diseases
5. **Implement severity assessment** for detected issues
6. **Add treatment recommendations** based on detected diseases

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

This project follows a modular development approach. Each component is designed to be independent and extensible. Contributions are welcome!

## 📞 Support

For questions or issues, please open an issue on GitHub or contact the development team.

---

**Built with ❤️ for sustainable agriculture and plant health monitoring**
