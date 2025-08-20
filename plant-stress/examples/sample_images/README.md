# Sample Images

This directory contains sample images for testing the plant stress detection system.

## Usage

You can use these images to test the model:

```python
from src.inference.plant_stress_detector import StablePlantDetector
import cv2

# Initialize detector
detector = StablePlantDetector("models/runs/classify/plantdoc_mobilevit_20250820_124432/best_model.pth")

# Load and test an image
image = cv2.imread("examples/sample_images/your_image.jpg")
predictions = detector.predict(image)

# Print results
for pred in predictions:
    print(f"{pred['class']}: {pred['confidence']}")
```

## Image Requirements

- **Format**: JPG, PNG, JPEG
- **Size**: Any size (will be resized to 224x224)
- **Color**: RGB/BGR format
- **Content**: Plant leaves (preferably close-up shots)

## Adding Your Own Images

1. Place your plant images in this directory
2. Use descriptive names (e.g., `tomato_leaf_blight.jpg`)
3. Test with the model using the code above

## Note

The original 3 images (Apple Scab.png, Blueberry.png, Peach.png) were removed during project cleanup. You can add your own plant images here for testing.
