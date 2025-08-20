#!/usr/bin/env python3
"""
Ablation Study: Compare MobileViT with EfficientNet-Lite models
Compare speed/accuracy trade-offs for plant stress detection
"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import timm
from timm.data import create_transform
from timm.utils import accuracy, AverageMeter

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

class ModelBenchmark:
    """Benchmark different models for speed and accuracy"""
    
    def __init__(self, data_dir, img_size=224, batch_size=32, device="auto"):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                     "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Create validation dataset
        val_dir = self.data_dir / "val"
        self.val_transform = create_transform(
            input_size=(img_size, img_size),
            is_training=False,
            interpolation='bicubic',
        )
        
        # Get classes from training data
        train_dir = self.data_dir / "train"
        self.classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
        self.num_classes = len(self.classes)
        
        print(f"Benchmarking {self.num_classes} classes on {self.device}")
    
    def create_model(self, model_name):
        """Create model with specified name"""
        print(f"Creating model: {model_name}")
        
        # Model configurations
        model_configs = {
            # MobileViT variants
            'mobilevit_xxs': 'mobilevit_xxs.cvnets_in1k',
            'mobilevit_xs': 'mobilevit_xs.cvnets_in1k',
            'mobilevit_s': 'mobilevit_s.cvnets_in1k',
            
            # EfficientNet-Lite variants
            'efficientnet_lite0': 'efficientnet_lite0.tf_imagenet',
            'efficientnet_lite1': 'efficientnet_lite1.tf_imagenet',
            'efficientnet_lite2': 'efficientnet_lite2.tf_imagenet',
            'efficientnet_lite3': 'efficientnet_lite3.tf_imagenet',
            'efficientnet_lite4': 'efficientnet_lite4.tf_imagenet',
            
            # Other efficient models for comparison
            'mobilenetv3_small_100': 'mobilenetv3_small_100.ra_in1k',
            'mobilenetv3_large_100': 'mobilenetv3_large_100.ra_in1k',
            'resnet18': 'resnet18.a1_in1k',
            'resnet50': 'resnet50.a1_in1k',
        }
        
        if model_name in model_configs:
            model = timm.create_model(
                model_configs[model_name],
                pretrained=True,
                num_classes=self.num_classes,
                drop_rate=0.1,
                drop_path_rate=0.1
            )
        else:
            # Try direct model name
            model = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=self.num_classes
            )
        
        return model
    
    def count_parameters(self, model):
        """Count model parameters"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params
    
    def measure_inference_speed(self, model, num_runs=100):
        """Measure inference speed"""
        model.eval()
        model = model.to(self.device)
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, self.img_size, self.img_size).to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Measure inference time
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        fps = 1.0 / avg_time
        
        return avg_time, std_time, fps
    
    def evaluate_model(self, model, val_loader):
        """Evaluate model accuracy"""
        model.eval()
        model = model.to(self.device)
        
        criterion = nn.CrossEntropyLoss()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = criterion(output, target)
                
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.update(loss.item(), data.size(0))
                top1.update(acc1.item(), data.size(0))
                top5.update(acc5.item(), data.size(0))
                
                # Store predictions
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        return losses.avg, top1.avg, top5.avg, all_preds, all_targets
    
    def benchmark_model(self, model_name):
        """Benchmark a single model"""
        print(f"\n🔍 Benchmarking {model_name}...")
        
        # Create model
        model = self.create_model(model_name)
        
        # Count parameters
        total_params, trainable_params = self.count_parameters(model)
        
        # Measure inference speed
        avg_time, std_time, fps = self.measure_inference_speed(model)
        
        # Create validation dataset and loader
        from PIL import Image
        
        class SimpleDataset:
            def __init__(self, data_dir, transform, classes):
                self.data_dir = Path(data_dir)
                self.transform = transform
                self.classes = classes
                self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
                
                self.images = []
                self.labels = []
                
                for class_name in self.classes:
                    class_dir = self.data_dir / class_name
                    class_idx = self.class_to_idx[class_name]
                    
                    for img_path in class_dir.glob("*.jpg"):
                        self.images.append(str(img_path))
                        self.labels.append(class_idx)
            
            def __len__(self):
                return len(self.images)
            
            def __getitem__(self, idx):
                img_path = self.images[idx]
                label = self.labels[idx]
                
                image = Image.open(img_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                return image, label
        
        val_dataset = SimpleDataset(
            self.data_dir / "val", 
            self.val_transform, 
            self.classes
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=2
        )
        
        # Evaluate accuracy
        loss, top1_acc, top5_acc, preds, targets = self.evaluate_model(model, val_loader)
        
        # Generate classification report
        report = classification_report(
            targets, preds, 
            target_names=self.classes, 
            output_dict=True
        )
        
        # Calculate macro F1 score
        macro_f1 = report['macro avg']['f1-score']
        
        results = {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'inference_time_ms': avg_time * 1000,
            'inference_time_std_ms': std_time * 1000,
            'fps': fps,
            'loss': loss,
            'top1_accuracy': top1_acc,
            'top5_accuracy': top5_acc,
            'macro_f1': macro_f1,
            'classification_report': report
        }
        
        print(f"  Parameters: {total_params:,} (trainable: {trainable_params:,})")
        print(f"  Inference: {avg_time*1000:.2f}ms ± {std_time*1000:.2f}ms ({fps:.1f} FPS)")
        print(f"  Top-1 Accuracy: {top1_acc:.2f}%")
        print(f"  Top-5 Accuracy: {top5_acc:.2f}%")
        print(f"  Macro F1: {macro_f1:.4f}")
        
        return results
    
    def run_ablation_study(self, models_to_test):
        """Run ablation study on multiple models"""
        print("🚀 Starting Ablation Study")
        print("=" * 60)
        
        results = []
        
        for model_name in models_to_test:
            try:
                result = self.benchmark_model(model_name)
                results.append(result)
            except Exception as e:
                print(f"❌ Failed to benchmark {model_name}: {e}")
                continue
        
        return results
    
    def plot_results(self, results, save_path):
        """Plot ablation study results"""
        if not results:
            print("No results to plot")
            return
        
        # Extract data
        model_names = [r['model_name'] for r in results]
        accuracies = [r['top1_accuracy'] for r in results]
        f1_scores = [r['macro_f1'] for r in results]
        fps_values = [r['fps'] for r in results]
        param_counts = [r['total_params'] for r in results]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Accuracy vs FPS
        ax1.scatter(fps_values, accuracies, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax1.annotate(name, (fps_values[i], accuracies[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('FPS (Inference Speed)')
        ax1.set_ylabel('Top-1 Accuracy (%)')
        ax1.set_title('Accuracy vs Speed Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # F1 Score vs Parameters
        ax2.scatter(param_counts, f1_scores, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax2.annotate(name, (param_counts[i], f1_scores[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Number of Parameters')
        ax2.set_ylabel('Macro F1 Score')
        ax2.set_title('F1 Score vs Model Size')
        ax2.grid(True, alpha=0.3)
        
        # Accuracy comparison
        bars1 = ax3.bar(range(len(model_names)), accuracies, alpha=0.7)
        ax3.set_xlabel('Models')
        ax3.set_ylabel('Top-1 Accuracy (%)')
        ax3.set_title('Accuracy Comparison')
        ax3.set_xticks(range(len(model_names)))
        ax3.set_xticklabels(model_names, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # FPS comparison
        bars2 = ax4.bar(range(len(model_names)), fps_values, alpha=0.7)
        ax4.set_xlabel('Models')
        ax4.set_ylabel('FPS')
        ax4.set_title('Speed Comparison')
        ax4.set_xticks(range(len(model_names)))
        ax4.set_xticklabels(model_names, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Results plotted to: {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Ablation study for plant stress detection models")
    parser.add_argument("--data-dir", default="data_proc", help="Data directory")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--output-dir", default="runs/ablation", help="Output directory")
    parser.add_argument("--models", nargs="+", default=[
        "mobilevit_xxs", "mobilevit_xs", "mobilevit_s",
        "efficientnet_lite0", "efficientnet_lite1", "efficientnet_lite2",
        "mobilenetv3_small_100", "mobilenetv3_large_100"
    ], help="Models to test")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Model Ablation Study")
    print("=" * 60)
    print(f"Testing models: {args.models}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"ablation_study_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create benchmark
    benchmark = ModelBenchmark(
        args.data_dir, 
        args.img_size, 
        args.batch_size, 
        args.device
    )
    
    # Run ablation study
    results = benchmark.run_ablation_study(args.models)
    
    if not results:
        print("❌ No models were successfully benchmarked")
        return
    
    # Save results
    results_file = output_dir / "ablation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary table
    print("\n📋 Ablation Study Results Summary")
    print("=" * 80)
    print(f"{'Model':<20} {'Params':<12} {'FPS':<8} {'Top-1':<8} {'F1':<8}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model_name']:<20} "
              f"{result['total_params']:<12,} "
              f"{result['fps']:<8.1f} "
              f"{result['top1_accuracy']:<8.2f} "
              f"{result['macro_f1']:<8.4f}")
    
    # Plot results
    plot_path = output_dir / "ablation_results.png"
    benchmark.plot_results(results, plot_path)
    
    # Find best models
    best_accuracy = max(results, key=lambda x: x['top1_accuracy'])
    best_f1 = max(results, key=lambda x: x['macro_f1'])
    fastest = max(results, key=lambda x: x['fps'])
    smallest = min(results, key=lambda x: x['total_params'])
    
    print(f"\n🏆 Best Models:")
    print(f"  Highest Accuracy: {best_accuracy['model_name']} ({best_accuracy['top1_accuracy']:.2f}%)")
    print(f"  Highest F1 Score: {best_f1['model_name']} ({best_f1['macro_f1']:.4f})")
    print(f"  Fastest: {fastest['model_name']} ({fastest['fps']:.1f} FPS)")
    print(f"  Smallest: {smallest['model_name']} ({smallest['total_params']:,} params)")
    
    print(f"\n✅ Ablation study completed!")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()
