#!/usr/bin/env python3
"""
Quick test to verify the model works
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import timm
from timm.data import create_transform

# Class names
CLASS_NAMES = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf', 
    'Bell_pepper leaf spot', 'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 
    'Corn leaf blight', 'Corn rust leaf', 'Peach leaf', 'Potato leaf early blight', 
    'Potato leaf late blight', 'Raspberry leaf', 'Soyabean leaf', 
    'Squash Powdery mildew leaf', 'Strawberry leaf', 'Tomato Early blight leaf', 
    'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 
    'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 
    'Tomato mold leaf', 'grape leaf', 'grape leaf black rot'
]

def test_model():
    """Test if the model loads and works"""
    print("🧪 Testing Plant Stress Detection Model...")
    
    # Model path
    model_path = "models/runs/classify/plantdoc_mobilevit_20250820_124432/best_model.pth"
    
    try:
        # Create model
        model = timm.create_model(
            'mobilevit_xxs.cvnets_in1k',
            pretrained=False,
            num_classes=len(CLASS_NAMES)
        )
        
        # Load weights
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("✅ Model loaded successfully!")
        
        # Create transform
        transform = create_transform(
            input_size=224,
            is_training=False,
            crop_pct=0.875,
            interpolation='bicubic'
        )
        
        # Create a dummy image (random noise)
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        pil_image = Image.fromarray(dummy_image)
        
        # Transform
        tensor = transform(pil_image).unsqueeze(0)
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = F.softmax(outputs, dim=1)
        
        # Get top predictions
        top_probs, top_indices = torch.topk(probabilities, 3)
        
        print("✅ Model inference works!")
        print("📊 Top 3 predictions:")
        for i in range(3):
            class_idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            class_name = CLASS_NAMES[class_idx]
            print(f"   {i+1}. {class_name}: {prob:.1%}")
        
        print("\n🎉 Model is ready for camera use!")
        print("💡 Run: python simple_mvp.py --camera 0")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    test_model()
