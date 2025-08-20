#!/usr/bin/env python3
"""
Phase 4: SageMaker MobileViT Training Script
MobileViT training with SMDDP support for hyperparameter tuning
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy
from timm.utils import ModelEma, accuracy, AverageMeter

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

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

def setup_distributed():
    """Setup distributed training with SMDDP"""
    if 'SM_HOSTS' in os.environ:
        # SageMaker distributed training
        hosts = json.loads(os.environ['SM_HOSTS'])
        current_host = os.environ['SM_CURRENT_HOST']
        rank = hosts.index(current_host)
        world_size = len(hosts)
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        print(f"✅ SMDDP initialized: rank={rank}, world_size={world_size}")
        return rank, world_size
    else:
        # Local training
        print("⚠️ Running in local mode (no distributed training)")
        return 0, 1

def cleanup_distributed():
    """Cleanup distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_model(model_name, num_classes, pretrained=True, task="classification"):
    """Create MobileViT model"""
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
            num_classes=1,
            drop_rate=0.1,
            drop_path_rate=0.1
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

def create_class_balanced_sampler(dataset):
    """Create class-balanced sampler"""
    class_counts = {}
    for label in dataset.labels:
        class_counts[label] = class_counts.get(label, 0) + 1
    
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
        else:  # regression
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
                
                preds = output.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            else:  # regression
                loss = criterion(output.squeeze(), target.float())
                mse = nn.MSELoss()(output.squeeze(), target.float())
                acc1 = 1.0 / (1.0 + mse.item())
                acc5 = acc1
                
                all_preds.extend(output.squeeze().cpu().numpy())
                all_targets.extend(target.cpu().numpy())
            
            losses.update(loss.item(), data.size(0))
            top1.update(acc1, data.size(0))
            top5.update(acc5, data.size(0))
    
    return losses.avg, top1.avg, top5.avg, all_preds, all_targets

def train_mobilevit_with_hyperparameters(hyperparameters, data_dir, model_dir):
    """Train MobileViT with given hyperparameters"""
    
    # Extract hyperparameters
    model_name = hyperparameters.get('model_name', 'mobilevit_xxs.cvnets_in1k')
    img_size = hyperparameters.get('img_size', 224)
    batch_size = hyperparameters.get('batch_size', 32)
    epochs = hyperparameters.get('epochs', 100)
    lr = hyperparameters.get('lr', 3e-4)
    weight_decay = hyperparameters.get('weight_decay', 1e-4)
    task = hyperparameters.get('task', 'classification')
    loss_type = hyperparameters.get('loss', 'ce')
    use_balanced_sampler = hyperparameters.get('use_balanced_sampler', False)
    use_ema = hyperparameters.get('use_ema', False)
    
    # Setup distributed training
    rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    
    try:
        # Create datasets
        train_dir = Path(data_dir) / "train"
        val_dir = Path(data_dir) / "val"
        
        if not train_dir.exists() or not val_dir.exists():
            print(f"❌ Data directories not found: {train_dir}, {val_dir}")
            return False
        
        # Create transforms
        train_transform = create_transforms(img_size, is_training=True)
        val_transform = create_transforms(img_size, is_training=False)
        
        # Create datasets
        train_dataset = PlantStressDataset(train_dir, transform=train_transform, is_training=True)
        val_dataset = PlantStressDataset(val_dir, transform=val_transform, is_training=False)
        
        num_classes = len(train_dataset.classes) if task == "classification" else 1
        print(f"Number of classes: {num_classes}")
        
        # Create data loaders
        if use_balanced_sampler and task == "classification":
            sampler = create_class_balanced_sampler(train_dataset)
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=sampler, 
                num_workers=4, pin_memory=True
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True, 
                num_workers=4, pin_memory=True
            )
        
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=4, pin_memory=True
        )
        
        # Create model
        model = create_model(model_name, num_classes, pretrained=True, task=task)
        model = model.to(device)
        
        # Wrap with DDP if distributed
        if world_size > 1:
            model = DDP(model, device_ids=[rank])
        
        # Create EMA model
        ema_model = None
        if use_ema:
            ema_model = ModelEma(model, decay=0.9999)
        
        # Create criterion
        if task == "classification":
            if loss_type == "ce":
                criterion = nn.CrossEntropyLoss()
            elif loss_type == "focal":
                criterion = timm.loss.FocalLoss()
            elif loss_type == "smooth":
                criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
        else:  # regression
            criterion = nn.MSELoss()
        
        # Create optimizer and scheduler
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # Setup TensorBoard
        if rank == 0:  # Only log on main process
            writer = SummaryWriter(log_dir=f"/tmp/tensorboard_logs")
        else:
            writer = None
        
        # Training loop
        best_acc = 0.0
        
        print(f"🚀 Starting MobileViT training with hyperparameters:")
        print(f"   Model: {model_name}")
        print(f"   Image size: {img_size}")
        print(f"   Batch size: {batch_size}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {lr}")
        print(f"   Task: {task}")
        print(f"   Loss: {loss_type}")
        print(f"   Rank: {rank}, World size: {world_size}")
        print()
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_acc5 = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch, ema_model, writer, task
            )
            
            # Validate
            val_model = ema_model.ema if ema_model else model
            val_loss, val_acc, val_acc5, val_preds, val_targets = validate(
                val_model, val_loader, criterion, device, task
            )
            
            # Update scheduler
            scheduler.step()
            
            # Log to TensorBoard
            if writer and rank == 0:
                writer.add_scalar('Loss/Val', val_loss, epoch)
                if task == "classification":
                    writer.add_scalar('Accuracy/Val_Top1', val_acc, epoch)
                    writer.add_scalar('Accuracy/Val_Top5', val_acc5, epoch)
                else:
                    writer.add_scalar('MSE/Val', 1.0/val_acc - 1.0, epoch)
                writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
            
            print(f"Epoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                if rank == 0:  # Only save on main process
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': val_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_acc': best_acc,
                        'hyperparameters': hyperparameters,
                    }, Path(model_dir) / "best_model.pth")
                    print(f"  🏆 New best accuracy: {best_acc:.2f}%")
        
        # Final evaluation
        if rank == 0:
            # Load best model
            checkpoint = torch.load(Path(model_dir) / "best_model.pth", map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            final_loss, final_acc, final_acc5, final_preds, final_targets = validate(
                model, val_loader, criterion, device, task
            )
            
            # Calculate metrics
            metrics = {}
            if task == "classification":
                report = classification_report(
                    final_targets, final_preds, 
                    target_names=train_dataset.classes, 
                    output_dict=True
                )
                metrics['macro_f1'] = report['macro avg']['f1-score']
                metrics['accuracy'] = final_acc
            else:  # regression
                mse = np.mean((np.array(final_preds) - np.array(final_targets)) ** 2)
                rmse = np.sqrt(mse)
                metrics['rmse'] = rmse
                metrics['mse'] = mse
            
            # Save metrics
            with open(Path(model_dir) / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"Final Results:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
        
        if writer:
            writer.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return False
    finally:
        cleanup_distributed()

def main():
    parser = argparse.ArgumentParser(description="Phase 4: SageMaker MobileViT Training")
    parser.add_argument("--hyperparameters", type=json.loads, default="{}", 
                       help="Hyperparameters as JSON string")
    parser.add_argument("--model-dir", default="/opt/ml/model", 
                       help="Model output directory")
    parser.add_argument("--data-dir", default="/opt/ml/input/data/training", 
                       help="Training data directory")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 4")
    print("SageMaker MobileViT Training")
    print("=" * 60)
    print(f"Model directory: {args.model_dir}")
    print(f"Data directory: {args.data_dir}")
    print(f"Hyperparameters: {args.hyperparameters}")
    print()
    
    # Check if running in SageMaker
    if 'SM_TRAINING_ENV' in os.environ:
        print("✅ Running in SageMaker environment")
        training_env = json.loads(os.environ['SM_TRAINING_ENV'])
        print(f"   Job name: {training_env.get('job_name', 'N/A')}")
        print(f"   Current host: {training_env.get('current_host', 'N/A')}")
    else:
        print("⚠️ Running in local environment")
    
    # Get dataset path
    if 'SM_CHANNEL_TRAINING' in os.environ:
        training_data_path = os.environ['SM_CHANNEL_TRAINING']
        print(f"📁 Training data path: {training_data_path}")
        data_dir = training_data_path
    else:
        print("⚠️ No training data channel found, using local data")
        data_dir = args.data_dir
    
    # Train model with hyperparameters
    success = train_mobilevit_with_hyperparameters(
        args.hyperparameters, 
        data_dir, 
        args.model_dir
    )
    
    if success:
        print("✅ Training completed successfully!")
    else:
        print("❌ Training failed!")

if __name__ == "__main__":
    main()
