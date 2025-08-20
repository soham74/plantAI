#!/usr/bin/env python3
"""
MobileViT Classifier Training for Plant Stress Detection
Stage B: Transformer classifier on crops for stress classification/severity regression
"""

import os
import sys
import argparse
import json
import yaml
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

import timm
from timm.data import create_transform
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import ModelEma, accuracy, AverageMeter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class PlantStressDataset:
    """Custom dataset for plant stress classification"""
    
    def __init__(self, data_dir, transform=None, is_training=True):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.is_training = is_training
        
        # Find all classes
        self.classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths and labels
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            
            for img_path in class_dir.glob("*.jpg"):
                self.images.append(str(img_path))
                self.labels.append(class_idx)
        
        print(f"Found {len(self.images)} images in {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def create_model(model_name, num_classes, pretrained=True):
    """Create MobileViT model"""
    print(f"Creating model: {model_name}")
    
    # Available MobileViT variants
    mobilevit_models = {
        'mobilevit_xxs': 'mobilevit_xxs.cvnets_in1k',
        'mobilevit_xs': 'mobilevit_xs.cvnets_in1k', 
        'mobilevit_s': 'mobilevit_s.cvnets_in1k'
    }
    
    if model_name in mobilevit_models:
        model = timm.create_model(
            mobilevit_models[model_name],
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
    else:
        # Fallback to other models for ablation studies
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes
        )
    
    return model

def create_transforms(img_size=224, is_training=True):
    """Create data transforms"""
    if is_training:
        transform = create_transform(
            input_size=(img_size, img_size),
            is_training=True,
            color_jitter=0.4,
            auto_augment='rand-m9-mstd0.5-inc1',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            interpolation='bicubic',
        )
    else:
        transform = create_transform(
            input_size=(img_size, img_size),
            is_training=False,
            interpolation='bicubic',
        )
    
    return transform

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, ema_model=None):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        if ema_model is not None:
            ema_model.update(model)
        
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        top5.update(acc5.item(), data.size(0))
        
        if batch_idx % 50 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(loader)}, '
                  f'Loss: {losses.avg:.4f}, Top1: {top1.avg:.2f}%, Top5: {top5.avg:.2f}%')
    
    return losses.avg, top1.avg, top5.avg

def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
            
            # Store predictions for detailed analysis
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    return losses.avg, top1.avg, top5.avg, all_preds, all_targets

def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc,
    }, save_path)

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Train MobileViT classifier for plant stress detection")
    parser.add_argument("--data-dir", default="data_proc", help="Data directory")
    parser.add_argument("--model", default="mobilevit_xxs", help="Model name")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--output-dir", default="runs/classify", help="Output directory")
    parser.add_argument("--experiment", default="mobilevit_plant_stress", help="Experiment name")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🌱 Plant Stress Detection - MobileViT Classifier Training")
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print("=" * 60)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.experiment}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['device'] = str(device)
    config['timestamp'] = timestamp
    
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create datasets
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        print(f"❌ Data directories not found: {train_dir}, {val_dir}")
        return
    
    # Create transforms
    train_transform = create_transforms(args.img_size, is_training=True)
    val_transform = create_transforms(args.img_size, is_training=False)
    
    # Create datasets
    train_dataset = PlantStressDataset(train_dir, transform=train_transform, is_training=True)
    val_dataset = PlantStressDataset(val_dir, transform=val_transform, is_training=False)
    
    num_classes = len(train_dataset.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Create model
    model = create_model(args.model, num_classes, pretrained=True)
    model = model.to(device)
    
    # Create EMA model
    ema_model = ModelEma(model, decay=0.9999)
    
    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = create_optimizer(args, model)
    
    # Create scheduler
    scheduler = create_scheduler(args, optimizer)
    
    # Training loop
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("🚀 Starting training...")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, ema_model
        )
        
        # Validate
        val_loss, val_acc, val_acc5, val_preds, val_targets = validate(
            ema_model.ema, val_loader, criterion, device
        )
        
        # Update scheduler
        if scheduler:
            scheduler.step(epoch)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                ema_model.ema, optimizer, scheduler, epoch, best_acc,
                output_dir / "best_model.pth"
            )
            print(f"  🏆 New best accuracy: {best_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_acc,
                output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            )
    
    # Final validation with best model
    print("🔍 Final validation with best model...")
    best_model = create_model(args.model, num_classes, pretrained=False)
    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    final_loss, final_acc, final_acc5, final_preds, final_targets = validate(
        best_model, val_loader, criterion, device
    )
    
    print(f"Final Results:")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Top-1 Accuracy: {final_acc:.2f}%")
    print(f"  Top-5 Accuracy: {final_acc5:.2f}%")
    
    # Generate detailed classification report
    report = classification_report(
        final_targets, final_preds, 
        target_names=train_dataset.classes, 
        output_dict=True
    )
    
    with open(output_dir / "classification_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        output_dir / "training_curves.png"
    )
    
    # Plot confusion matrix
    plot_confusion_matrix(
        final_targets, final_preds, train_dataset.classes,
        output_dir / "confusion_matrix.png"
    )
    
    # Save final model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'config': config,
        'classes': train_dataset.classes,
        'final_accuracy': final_acc,
        'best_accuracy': best_acc,
    }, output_dir / "final_model.pth")
    
    print(f"✅ Training completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Best accuracy: {best_acc:.2f}%")

if __name__ == "__main__":
    main()
