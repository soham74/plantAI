#!/usr/bin/env python3
"""
Phase 3B: Crop Classifier (PyTorch, timm MobileViT)
MobileViT classifier for plant stress classification and severity regression
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
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from torchvision import datasets

import timm
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import ModelEma, accuracy, AverageMeter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
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

def create_model(model_name, num_classes, pretrained=True, task="classification"):
    """
    Create MobileViT model with specified configuration
    
    Args:
        model_name: Model name (e.g., 'mobilevit_xxs.cvnets_in1k')
        num_classes: Number of classes
        pretrained: Whether to use pretrained weights
        task: 'classification' or 'regression'
    """
    print(f"Creating model: {model_name}")
    
    if task == "classification":
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_rate=0.1,
            drop_path_rate=0.1
        )
    else:  # regression
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=1,  # Single output for regression
            drop_rate=0.1,
            drop_path_rate=0.1
        )
    
    return model

def create_transforms(img_size=224, is_training=True):
    """
    Create data transforms with Phase 3B specifications
    
    Args:
        img_size: Input image size (224 or 256)
        is_training: Whether transforms are for training
    """
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

def create_class_balanced_sampler(dataset):
    """Create class-balanced sampler for skewed datasets"""
    class_counts = {}
    for label in dataset.labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    # Compute class weights
    class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
    sample_weights = [class_weights[label] for label in dataset.labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, 
                   ema_model=None, writer=None, task="classification"):
    """Train for one epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        if task == "classification":
            loss = criterion(output, target)
        else:  # regression
            loss = criterion(output.squeeze(), target.float())
        
        loss.backward()
        optimizer.step()
        
        if ema_model is not None:
            ema_model.update(model)
        
        if task == "classification":
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1.item(), data.size(0))
            top5.update(acc5.item(), data.size(0))
        else:  # regression - use MSE as accuracy proxy
            mse = nn.MSELoss()(output.squeeze(), target.float())
            top1.update(1.0 / (1.0 + mse.item()), data.size(0))
            top5.update(1.0 / (1.0 + mse.item()), data.size(0))
        
        losses.update(loss.item(), data.size(0))
        
        # Log to TensorBoard
        if writer and batch_idx % 50 == 0:
            step = epoch * len(loader) + batch_idx
            writer.add_scalar('Loss/Train', loss.item(), step)
            if task == "classification":
                writer.add_scalar('Accuracy/Train_Top1', acc1.item(), step)
                writer.add_scalar('Accuracy/Train_Top5', acc5.item(), step)
            else:
                writer.add_scalar('MSE/Train', mse.item(), step)
        
        if batch_idx % 50 == 0:
            if task == "classification":
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(loader)}, '
                      f'Loss: {losses.avg:.4f}, Top1: {top1.avg:.2f}%, Top5: {top5.avg:.2f}%')
            else:
                print(f'Epoch: {epoch}, Batch: {batch_idx}/{len(loader)}, '
                      f'Loss: {losses.avg:.4f}, MSE: {1.0/top1.avg - 1.0:.4f}')
    
    return losses.avg, top1.avg, top5.avg

def validate(model, loader, criterion, device, task="classification"):
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
            
            if task == "classification":
                loss = criterion(output, target)
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                
                # Store predictions for detailed analysis
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            else:  # regression
                loss = criterion(output.squeeze(), target.float())
                mse = nn.MSELoss()(output.squeeze(), target.float())
                acc1 = 1.0 / (1.0 + mse.item())
                acc5 = acc1
                
                # Store predictions for regression analysis
                all_preds.extend(output.squeeze().cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
            top5.update(acc5, data.size(0))
    
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

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path, task="classification"):
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
    if task == "classification":
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_ylabel('Accuracy (%)')
    else:
        ax2.plot(train_accs, label='Train (1/(1+MSE))')
        ax2.plot(val_accs, label='Val (1/(1+MSE))')
        ax2.set_title('Training and Validation Performance')
        ax2.set_ylabel('1/(1+MSE)')
    
    ax2.set_xlabel('Epoch')
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
    parser = argparse.ArgumentParser(description="Phase 3B: MobileViT Crop Classifier Training")
    parser.add_argument("--data-dir", default="data/data_proc", help="Data directory")
    parser.add_argument("--model", default="mobilevit_xxs.cvnets_in1k", help="Model name")
    parser.add_argument("--img-size", type=int, default=224, choices=[224, 256], help="Input image size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--output-dir", default="runs/classify", help="Output directory")
    parser.add_argument("--experiment", default="mobilevit_plant_stress", help="Experiment name")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification", 
                       help="Task type")
    parser.add_argument("--loss", choices=["ce", "focal", "smooth"], default="ce", help="Loss function")
    parser.add_argument("--use-balanced-sampler", action="store_true", help="Use class-balanced sampler")
    parser.add_argument("--use-ema", action="store_true", help="Use EMA model")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else 
                            "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print("🌱 Plant Stress Detection - Phase 3B")
    print("MobileViT Crop Classifier Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Image size: {args.img_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Task: {args.task}")
    print(f"Loss: {args.loss}")
    print()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.experiment}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir=output_dir / "tensorboard")
    
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
    
    num_classes = len(train_dataset.classes) if args.task == "classification" else 1
    print(f"Number of classes: {num_classes}")
    if args.task == "classification":
        print(f"Classes: {train_dataset.classes}")
    
    # Create data loaders
    if args.use_balanced_sampler and args.task == "classification":
        sampler = create_class_balanced_sampler(train_dataset)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=sampler, 
            num_workers=4, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True, 
            num_workers=4, pin_memory=True
        )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )
    
    # Create model (exactly as specified in the loop sketch)
    net = create_model(args.model, num_classes, pretrained=True, task=args.task)
    net = net.to(device)
    
    # Create EMA model if requested
    ema_model = None
    if args.use_ema:
        ema_model = ModelEma(net, decay=0.9999)
    
    # Create criterion
    if args.task == "classification":
        if args.loss == "ce":
            criterion = nn.CrossEntropyLoss()
        elif args.loss == "focal":
            criterion = timm.loss.FocalLoss()
        elif args.loss == "smooth":
            criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    else:  # regression
        criterion = nn.MSELoss()
    
    # Create optimizer (exactly as specified)
    opt = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Create scheduler (exactly as specified)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    
    # Training loop (standard PyTorch train loop as specified)
    best_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print("🚀 Starting training...")
    
    for epoch in range(args.epochs):
        # Train
        train_loss, train_acc, train_acc5 = train_one_epoch(
            net, train_loader, criterion, opt, device, epoch, ema_model, writer, args.task
        )
        
        # Validate
        val_model = ema_model.ema if ema_model else net
        val_loss, val_acc, val_acc5, val_preds, val_targets = validate(
            val_model, val_loader, criterion, device, args.task
        )
        
        # Update scheduler
        sched.step()
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Log to TensorBoard
        writer.add_scalar('Loss/Val', val_loss, epoch)
        if args.task == "classification":
            writer.add_scalar('Accuracy/Val_Top1', val_acc, epoch)
            writer.add_scalar('Accuracy/Val_Top5', val_acc5, epoch)
        else:
            writer.add_scalar('MSE/Val', 1.0/val_acc - 1.0, epoch)
        writer.add_scalar('Learning_Rate', sched.get_last_lr()[0], epoch)
        
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                val_model, opt, sched, epoch, best_acc,
                output_dir / "best_model.pth"
            )
            print(f"  🏆 New best accuracy: {best_acc:.2f}%")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                net, opt, sched, epoch, best_acc,
                output_dir / f"checkpoint_epoch_{epoch+1}.pth"
            )
    
    # Final validation with best model
    print("🔍 Final validation with best model...")
    best_model = create_model(args.model, num_classes, pretrained=False, task=args.task)
    checkpoint = torch.load(output_dir / "best_model.pth", map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    best_model = best_model.to(device)
    
    final_loss, final_acc, final_acc5, final_preds, final_targets = validate(
        best_model, val_loader, criterion, device, args.task
    )
    
    print(f"Final Results:")
    print(f"  Loss: {final_loss:.4f}")
    if args.task == "classification":
        print(f"  Top-1 Accuracy: {final_acc:.2f}%")
        print(f"  Top-5 Accuracy: {final_acc5:.2f}%")
    else:
        print(f"  MSE: {1.0/final_acc - 1.0:.4f}")
    
    # Generate detailed classification report
    if args.task == "classification":
        report = classification_report(
            final_targets, final_preds, 
            target_names=train_dataset.classes, 
            output_dict=True
        )
        
        with open(output_dir / "classification_report.json", 'w') as f:
            json.dump(report, f, indent=2)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            final_targets, final_preds, train_dataset.classes,
            output_dir / "confusion_matrix.png"
        )
    
    # Plot training curves
    plot_training_curves(
        train_losses, val_losses, train_accs, val_accs,
        output_dir / "training_curves.png", args.task
    )
    
    # Save final model
    torch.save({
        'model_state_dict': best_model.state_dict(),
        'config': config,
        'classes': train_dataset.classes if args.task == "classification" else None,
        'final_accuracy': final_acc,
        'best_accuracy': best_acc,
    }, output_dir / "final_model.pth")
    
    # Close TensorBoard writer
    writer.close()
    
    print(f"✅ Training completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Best accuracy: {best_acc:.2f}%")
    print(f"TensorBoard logs: {output_dir}/tensorboard")

if __name__ == "__main__":
    main()
