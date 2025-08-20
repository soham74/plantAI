# Plant Stress Detection API Reference

## Core Classes

### StablePlantDetector

Main class for plant stress detection using MobileViT model.

#### Constructor
```python
StablePlantDetector(model_path, device="auto")
```

**Parameters:**
- `model_path` (str): Path to the trained model file
- `device` (str): Device to use ("auto", "cpu", "mps", "cuda")

#### Methods

##### predict(image)
Predict plant stress from an image.

**Parameters:**
- `image` (numpy.ndarray): Input image in BGR format

**Returns:**
- `list`: List of dictionaries with predictions
  ```python
  [
    {
      'class': 'Tomato leaf blight',
      'probability': 0.85,
      'confidence': '85.0%'
    },
    # ... more predictions
  ]
  ```

##### draw_predictions(image, predictions)
Draw predictions overlay on an image.

**Parameters:**
- `image` (numpy.ndarray): Input image
- `predictions` (list): List of prediction dictionaries

**Returns:**
- `numpy.ndarray`: Image with predictions drawn

## Utility Functions

### test_model()
Test if the model loads and works correctly.

**Returns:**
- `bool`: True if model works, False otherwise

## Usage Examples

### Basic Detection
```python
from src.inference.plant_stress_detector import StablePlantDetector

# Initialize detector
detector = StablePlantDetector("models/runs/classify/plantdoc_mobilevit_20250820_124432/best_model.pth")

# Load image
image = cv2.imread("plant_image.jpg")

# Get predictions
predictions = detector.predict(image)

# Draw results
result_image = detector.draw_predictions(image, predictions)
```

### Camera Detection
```python
# Run camera detection
python run_detector.py --camera 0
```

## Model Information

- **Architecture**: MobileViT-XXS
- **Input Size**: 224x224 RGB
- **Classes**: 27 plant stress types
- **Accuracy**: 51.48%
- **Speed**: Real-time on Apple Silicon

## Supported Plant Stress Types

1. Apple Scab Leaf
2. Apple leaf
3. Apple rust leaf
4. Bell_pepper leaf
5. Bell_pepper leaf spot
6. Blueberry leaf
7. Cherry leaf
8. Corn Gray leaf spot
9. Corn leaf blight
10. Corn rust leaf
11. Peach leaf
12. Potato leaf early blight
13. Potato leaf late blight
14. Raspberry leaf
15. Soyabean leaf
16. Squash Powdery mildew leaf
17. Strawberry leaf
18. Tomato Early blight leaf
19. Tomato Septoria leaf spot
20. Tomato leaf
21. Tomato leaf bacterial spot
22. Tomato leaf late blight
23. Tomato leaf mosaic virus
24. Tomato leaf yellow virus
25. Tomato mold leaf
26. grape leaf
27. grape leaf black rot
