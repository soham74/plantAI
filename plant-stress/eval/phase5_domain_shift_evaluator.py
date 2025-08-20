#!/usr/bin/env python3
"""
Phase 5: Domain Shift Evaluator
Specialized evaluation for domain shift analysis between PlantVillage and phone datasets
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, f1_score, accuracy_score
)

import timm
from timm.data import create_transform

class PlantStressDataset(Dataset):
    """Custom dataset for plant stress classification"""
    
    def __init__(self, data_dir, transform=None, is_training=False):
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

class DomainShiftEvaluator:
    """Specialized evaluator for domain shift analysis"""
    
    def __init__(self, output_dir="domain_shift_results", tensorboard_dir="runs/domain_shift"):
        self.output_dir = Path(output_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        
        # Results storage
        self.results = {
            'plantvillage_model': {},
            'phone_model': {},
            'domain_shift_analysis': {},
            'failure_cases': {
                'plantvillage': [],
                'phone': []
            }
        }
        
        # Color palette
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))
    
    def load_model(self, model_path, model_name="mobilevit_xxs.cvnets_in1k", num_classes=None):
        """Load trained model"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            model_state = checkpoint['model_state_dict']
            config = checkpoint.get('config', {})
            classes = checkpoint.get('classes', None)
        else:
            model_state = checkpoint.state_dict()
            config = {}
            classes = None
        
        # Create model
        if num_classes is None and classes is not None:
            num_classes = len(classes)
        elif num_classes is None:
            num_classes = 27  # Default for PlantDoc
        
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        model.load_state_dict(model_state)
        model = model.to(device)
        model.eval()
        
        return model, classes, device
    
    def create_data_loader(self, data_dir, batch_size=32, img_size=224):
        """Create data loader for evaluation"""
        # Create transforms
        transform = create_transform(
            input_size=(img_size, img_size),
            is_training=False,
            interpolation='bicubic',
        )
        
        # Create dataset
        dataset = PlantStressDataset(data_dir, transform=transform, is_training=False)
        
        # Create data loader
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        
        return loader, dataset.classes
    
    def evaluate_model_on_dataset(self, model, data_loader, device, model_name="model"):
        """Evaluate model on a specific dataset"""
        print(f"🔍 Evaluating {model_name}...")
        
        all_preds = []
        all_targets = []
        all_probabilities = []
        failure_cases = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                probabilities = torch.softmax(output, dim=1)
                preds = output.argmax(dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                
                # Collect failure cases
                incorrect_mask = preds != target
                if incorrect_mask.any():
                    for i in range(len(data)):
                        if incorrect_mask[i]:
                            failure_cases.append({
                                'image': data[i].cpu(),
                                'true_label': target[i].cpu().item(),
                                'predicted_label': preds[i].cpu().item(),
                                'confidence': probabilities[i].max().cpu().item(),
                                'batch_idx': batch_idx,
                                'sample_idx': i
                            })
        
        # Calculate metrics
        metrics = self._calculate_comprehensive_metrics(all_targets, all_preds, all_probabilities)
        
        return metrics, failure_cases, all_targets, all_preds, all_probabilities
    
    def _calculate_comprehensive_metrics(self, targets, preds, probabilities):
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(targets, preds)
        metrics['macro_f1'] = f1_score(targets, preds, average='macro')
        metrics['weighted_f1'] = f1_score(targets, preds, average='weighted')
        metrics['micro_f1'] = f1_score(targets, preds, average='micro')
        
        # Per-class metrics
        report = classification_report(targets, preds, output_dict=True)
        for class_name, class_metrics in report.items():
            if isinstance(class_metrics, dict):
                metrics[f'f1_{class_name}'] = class_metrics.get('f1-score', 0)
                metrics[f'precision_{class_name}'] = class_metrics.get('precision', 0)
                metrics[f'recall_{class_name}'] = class_metrics.get('recall', 0)
        
        # Confusion matrix
        cm = confusion_matrix(targets, preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Average precision for each class
        num_classes = len(np.unique(targets))
        for i in range(num_classes):
            binary_targets = [1 if t == i else 0 for t in targets]
            ap = average_precision_score(binary_targets, probabilities[:, i])
            metrics[f'average_precision_class_{i}'] = ap
        
        return metrics
    
    def evaluate_domain_shift(self, plantvillage_model_path, phone_model_path, 
                            fgvc_data_dir, phone_data_dir, output_dir=None):
        """Evaluate domain shift between PlantVillage and phone datasets"""
        print("🌍 Evaluating Domain Shift Analysis...")
        print("=" * 60)
        
        # Load models
        plantvillage_model, plantvillage_classes, device = self.load_model(plantvillage_model_path)
        phone_model, phone_classes, device = self.load_model(phone_model_path)
        
        # Create data loaders
        fgvc_loader, fgvc_classes = self.create_data_loader(fgvc_data_dir)
        phone_loader, phone_classes = self.create_data_loader(phone_data_dir)
        
        # Evaluate PlantVillage model on FGVC data
        print("\n📊 Evaluating PlantVillage model on FGVC data...")
        plantvillage_fgvc_metrics, plantvillage_fgvc_failures, _, _, _ = self.evaluate_model_on_dataset(
            plantvillage_model, fgvc_loader, device, "PlantVillage model on FGVC"
        )
        
        # Evaluate PlantVillage model on phone data
        print("\n📊 Evaluating PlantVillage model on phone data...")
        plantvillage_phone_metrics, plantvillage_phone_failures, _, _, _ = self.evaluate_model_on_dataset(
            plantvillage_model, phone_loader, device, "PlantVillage model on phone"
        )
        
        # Evaluate phone model on FGVC data
        print("\n📊 Evaluating phone model on FGVC data...")
        phone_fgvc_metrics, phone_fgvc_failures, _, _, _ = self.evaluate_model_on_dataset(
            phone_model, fgvc_loader, device, "Phone model on FGVC"
        )
        
        # Evaluate phone model on phone data
        print("\n📊 Evaluating phone model on phone data...")
        phone_phone_metrics, phone_phone_failures, _, _, _ = self.evaluate_model_on_dataset(
            phone_model, phone_loader, device, "Phone model on phone"
        )
        
        # Store results
        self.results['plantvillage_model'] = {
            'fgvc_metrics': plantvillage_fgvc_metrics,
            'phone_metrics': plantvillage_phone_metrics,
            'fgvc_failures': plantvillage_fgvc_failures,
            'phone_failures': plantvillage_phone_failures
        }
        
        self.results['phone_model'] = {
            'fgvc_metrics': phone_fgvc_metrics,
            'phone_metrics': phone_phone_metrics,
            'fgvc_failures': phone_fgvc_failures,
            'phone_failures': phone_phone_failures
        }
        
        # Calculate domain shift deltas
        domain_shift_analysis = self._calculate_domain_shift_deltas(
            plantvillage_fgvc_metrics, plantvillage_phone_metrics,
            phone_fgvc_metrics, phone_phone_metrics
        )
        
        self.results['domain_shift_analysis'] = domain_shift_analysis
        
        # Generate visualizations
        self._generate_domain_shift_visualizations(domain_shift_analysis)
        
        # Save failure cases
        self._save_failure_cases()
        
        # Generate comprehensive report
        self._generate_domain_shift_report()
        
        print("\n✅ Domain shift evaluation completed!")
        return domain_shift_analysis
    
    def _calculate_domain_shift_deltas(self, plantvillage_fgvc, plantvillage_phone, 
                                     phone_fgvc, phone_phone):
        """Calculate domain shift deltas and analysis"""
        analysis = {
            'plantvillage_domain_shift': {},
            'phone_domain_adaptation': {},
            'cross_domain_comparison': {},
            'recommendations': []
        }
        
        # Calculate PlantVillage model domain shift (FGVC -> Phone)
        for key in plantvillage_fgvc:
            if isinstance(plantvillage_fgvc[key], (int, float)):
                delta = plantvillage_phone[key] - plantvillage_fgvc[key]
                analysis['plantvillage_domain_shift'][key] = {
                    'fgvc': plantvillage_fgvc[key],
                    'phone': plantvillage_phone[key],
                    'delta': delta,
                    'percent_change': (delta / plantvillage_fgvc[key] * 100) if plantvillage_fgvc[key] != 0 else 0
                }
        
        # Calculate phone model domain adaptation improvement
        for key in plantvillage_fgvc:
            if isinstance(plantvillage_fgvc[key], (int, float)):
                plantvillage_delta = plantvillage_phone[key] - plantvillage_fgvc[key]
                phone_delta = phone_phone[key] - phone_fgvc[key]
                improvement = phone_delta - plantvillage_delta
                
                analysis['phone_domain_adaptation'][key] = {
                    'plantvillage_delta': plantvillage_delta,
                    'phone_delta': phone_delta,
                    'improvement': improvement,
                    'improvement_percent': (improvement / abs(plantvillage_delta) * 100) if plantvillage_delta != 0 else 0
                }
        
        # Cross-domain comparison
        analysis['cross_domain_comparison'] = {
            'plantvillage_vs_phone_on_fgvc': {},
            'plantvillage_vs_phone_on_phone': {}
        }
        
        for key in plantvillage_fgvc:
            if isinstance(plantvillage_fgvc[key], (int, float)):
                # On FGVC data
                fgvc_diff = phone_fgvc[key] - plantvillage_fgvc[key]
                analysis['cross_domain_comparison']['plantvillage_vs_phone_on_fgvc'][key] = fgvc_diff
                
                # On phone data
                phone_diff = phone_phone[key] - plantvillage_phone[key]
                analysis['cross_domain_comparison']['plantvillage_vs_phone_on_phone'][key] = phone_diff
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_domain_shift_recommendations(analysis)
        
        return analysis
    
    def _generate_domain_shift_recommendations(self, analysis):
        """Generate recommendations based on domain shift analysis"""
        recommendations = []
        
        # Analyze PlantVillage domain shift
        plantvillage_shift = analysis['plantvillage_domain_shift']
        if plantvillage_shift:
            avg_delta = np.mean([v['delta'] for v in plantvillage_shift.values()])
            avg_percent_change = np.mean([v['percent_change'] for v in plantvillage_shift.values()])
            
            if avg_delta < -0.15:
                recommendations.append("Severe domain shift detected - PlantVillage model performs poorly on phone data")
                recommendations.append("Consider extensive domain adaptation or retraining on phone-like data")
            elif avg_delta < -0.05:
                recommendations.append("Moderate domain shift - some performance degradation on phone data")
                recommendations.append("Consider fine-tuning on phone dataset or data augmentation")
            else:
                recommendations.append("Minimal domain shift - PlantVillage model generalizes well to phone data")
        
        # Analyze phone model adaptation
        phone_adaptation = analysis['phone_domain_adaptation']
        if phone_adaptation:
            avg_improvement = np.mean([v['improvement'] for v in phone_adaptation.values()])
            
            if avg_improvement > 0.1:
                recommendations.append("Phone model shows significant improvement over PlantVillage model")
                recommendations.append("Phone-adapted training is highly effective for this domain")
            elif avg_improvement > 0.05:
                recommendations.append("Phone model shows moderate improvement over PlantVillage model")
                recommendations.append("Phone-adapted training provides some benefit")
            else:
                recommendations.append("Phone model shows minimal improvement over PlantVillage model")
                recommendations.append("Consider if phone-adapted training is necessary")
        
        # Cross-domain analysis
        cross_comparison = analysis['cross_domain_comparison']
        if cross_comparison:
            fgvc_comparison = cross_comparison['plantvillage_vs_phone_on_fgvc']
            phone_comparison = cross_comparison['plantvillage_vs_phone_on_phone']
            
            if fgvc_comparison:
                avg_fgvc_diff = np.mean(list(fgvc_comparison.values()))
                if avg_fgvc_diff < -0.05:
                    recommendations.append("Phone model underperforms on FGVC data - potential overfitting to phone domain")
                elif avg_fgvc_diff > 0.05:
                    recommendations.append("Phone model outperforms on FGVC data - good generalization")
            
            if phone_comparison:
                avg_phone_diff = np.mean(list(phone_comparison.values()))
                if avg_phone_diff > 0.1:
                    recommendations.append("Phone model significantly outperforms on phone data - effective domain adaptation")
                elif avg_phone_diff < 0.05:
                    recommendations.append("Phone model shows minimal improvement on phone data")
        
        return recommendations
    
    def _generate_domain_shift_visualizations(self, analysis):
        """Generate comprehensive domain shift visualizations"""
        print("📊 Generating domain shift visualizations...")
        
        # 1. Domain Shift Comparison
        self._plot_domain_shift_comparison(analysis)
        
        # 2. Model Performance Heatmap
        self._plot_model_performance_heatmap(analysis)
        
        # 3. Delta Analysis
        self._plot_delta_analysis(analysis)
        
        # 4. Cross-Domain Comparison
        self._plot_cross_domain_comparison(analysis)
        
        # 5. Log to TensorBoard
        self._log_domain_shift_tensorboard(analysis)
    
    def _plot_domain_shift_comparison(self, analysis):
        """Plot domain shift comparison"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Domain Shift Analysis: Model Performance Comparison', fontsize=16)
        
        # Extract key metrics
        metrics = ['accuracy', 'macro_f1', 'weighted_f1']
        
        # PlantVillage model performance
        plantvillage_shift = analysis['plantvillage_domain_shift']
        fgvc_values = [plantvillage_shift.get(m, {}).get('fgvc', 0) for m in metrics]
        phone_values = [plantvillage_shift.get(m, {}).get('phone', 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, fgvc_values, width, label='FGVC Data', alpha=0.8, color='skyblue')
        ax1.bar(x + width/2, phone_values, width, label='Phone Data', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('PlantVillage Model Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Phone model performance
        phone_adaptation = analysis['phone_domain_adaptation']
        phone_fgvc_values = [phone_adaptation.get(m, {}).get('phone_delta', 0) + plantvillage_shift.get(m, {}).get('fgvc', 0) for m in metrics]
        phone_phone_values = [phone_adaptation.get(m, {}).get('phone_delta', 0) + plantvillage_shift.get(m, {}).get('phone', 0) for m in metrics]
        
        ax2.bar(x - width/2, phone_fgvc_values, width, label='FGVC Data', alpha=0.8, color='skyblue')
        ax2.bar(x + width/2, phone_phone_values, width, label='Phone Data', alpha=0.8, color='lightcoral')
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Score')
        ax2.set_title('Phone Model Performance')
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Domain shift deltas
        deltas = [plantvillage_shift.get(m, {}).get('delta', 0) for m in metrics]
        colors = ['red' if d < 0 else 'green' for d in deltas]
        
        bars = ax3.bar(metrics, deltas, color=colors, alpha=0.7)
        ax3.set_xlabel('Metrics')
        ax3.set_ylabel('Delta (Phone - FGVC)')
        ax3.set_title('PlantVillage Model: Domain Shift Deltas')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{delta:+.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Phone model improvements
        improvements = [phone_adaptation.get(m, {}).get('improvement', 0) for m in metrics]
        colors = ['green' if i > 0 else 'red' for i in improvements]
        
        bars = ax4.bar(metrics, improvements, color=colors, alpha=0.7)
        ax4.set_xlabel('Metrics')
        ax4.set_ylabel('Improvement')
        ax4.set_title('Phone Model: Domain Adaptation Improvements')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, improvement in zip(bars, improvements):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{improvement:+.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_shift_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('Domain_Shift/Comparison', fig, 0)
    
    def _plot_model_performance_heatmap(self, analysis):
        """Plot model performance heatmap"""
        # Create performance matrix
        metrics = ['accuracy', 'macro_f1', 'weighted_f1']
        models = ['PlantVillage', 'Phone']
        datasets = ['FGVC', 'Phone']
        
        performance_matrix = np.zeros((len(metrics), len(models) * len(datasets)))
        
        # Fill matrix with performance values
        for i, metric in enumerate(metrics):
            # PlantVillage model
            plantvillage_shift = analysis['plantvillage_domain_shift']
            performance_matrix[i, 0] = plantvillage_shift.get(metric, {}).get('fgvc', 0)  # PlantVillage on FGVC
            performance_matrix[i, 1] = plantvillage_shift.get(metric, {}).get('phone', 0)  # PlantVillage on Phone
            
            # Phone model
            phone_adaptation = analysis['phone_domain_adaptation']
            performance_matrix[i, 2] = phone_adaptation.get(metric, {}).get('phone_delta', 0) + plantvillage_shift.get(metric, {}).get('fgvc', 0)  # Phone on FGVC
            performance_matrix[i, 3] = phone_adaptation.get(metric, {}).get('phone_delta', 0) + plantvillage_shift.get(metric, {}).get('phone', 0)  # Phone on Phone
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        
        # Set labels
        ax.set_xticks(range(len(models) * len(datasets)))
        ax.set_xticklabels(['PV-FGVC', 'PV-Phone', 'Phone-FGVC', 'Phone-Phone'])
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Performance Score')
        
        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(models) * len(datasets)):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.3f}',
                             ha="center", va="center", color="black", fontweight='bold')
        
        ax.set_title('Model Performance Heatmap\n(PV=PlantVillage, FGVC=FGVC Dataset, Phone=Phone Dataset)')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_performance_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('Domain_Shift/Performance_Heatmap', fig, 0)
    
    def _plot_delta_analysis(self, analysis):
        """Plot detailed delta analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Domain shift deltas
        plantvillage_shift = analysis['plantvillage_domain_shift']
        metrics = list(plantvillage_shift.keys())
        deltas = [plantvillage_shift[m]['delta'] for m in metrics if isinstance(plantvillage_shift[m]['delta'], (int, float))]
        percent_changes = [plantvillage_shift[m]['percent_change'] for m in metrics if isinstance(plantvillage_shift[m]['percent_change'], (int, float))]
        
        # Filter out non-numeric metrics
        numeric_metrics = [m for m in metrics if isinstance(plantvillage_shift[m]['delta'], (int, float))]
        
        colors = ['red' if d < 0 else 'green' for d in deltas]
        
        bars1 = ax1.bar(range(len(deltas)), deltas, color=colors, alpha=0.7)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Delta (Phone - FGVC)')
        ax1.set_title('PlantVillage Model: Domain Shift Deltas')
        ax1.set_xticks(range(len(deltas)))
        ax1.set_xticklabels(numeric_metrics, rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, delta in zip(bars1, deltas):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{delta:+.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # Percent changes
        colors = ['red' if p < 0 else 'green' for p in percent_changes]
        
        bars2 = ax2.bar(range(len(percent_changes)), percent_changes, color=colors, alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Percent Change (%)')
        ax2.set_title('PlantVillage Model: Domain Shift Percent Changes')
        ax2.set_xticks(range(len(percent_changes)))
        ax2.set_xticklabels(numeric_metrics, rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, pct in zip(bars2, percent_changes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (1 if height >= 0 else -2),
                    f'{pct:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'delta_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('Domain_Shift/Delta_Analysis', fig, 0)
    
    def _plot_cross_domain_comparison(self, analysis):
        """Plot cross-domain comparison"""
        cross_comparison = analysis['cross_domain_comparison']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # On FGVC data
        fgvc_comparison = cross_comparison['plantvillage_vs_phone_on_fgvc']
        metrics = list(fgvc_comparison.keys())
        fgvc_diffs = list(fgvc_comparison.values())
        
        colors = ['red' if d < 0 else 'green' for d in fgvc_diffs]
        
        bars1 = ax1.bar(range(len(fgvc_diffs)), fgvc_diffs, color=colors, alpha=0.7)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Difference (Phone - PlantVillage)')
        ax1.set_title('Model Comparison on FGVC Data')
        ax1.set_xticks(range(len(fgvc_diffs)))
        ax1.set_xticklabels(metrics, rotation=45)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, diff in zip(bars1, fgvc_diffs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{diff:+.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        # On phone data
        phone_comparison = cross_comparison['plantvillage_vs_phone_on_phone']
        phone_diffs = list(phone_comparison.values())
        
        colors = ['red' if d < 0 else 'green' for d in phone_diffs]
        
        bars2 = ax2.bar(range(len(phone_diffs)), phone_diffs, color=colors, alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Difference (Phone - PlantVillage)')
        ax2.set_title('Model Comparison on Phone Data')
        ax2.set_xticks(range(len(phone_diffs)))
        ax2.set_xticklabels(metrics, rotation=45)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, diff in zip(bars2, phone_diffs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{diff:+.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'cross_domain_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('Domain_Shift/Cross_Domain_Comparison', fig, 0)
    
    def _log_domain_shift_tensorboard(self, analysis):
        """Log domain shift analysis to TensorBoard"""
        # Log key metrics
        plantvillage_shift = analysis['plantvillage_domain_shift']
        for metric, data in plantvillage_shift.items():
            if isinstance(data, dict):
                self.writer.add_scalar(f'Domain_Shift/PlantVillage_{metric}_FGVC', data.get('fgvc', 0), 0)
                self.writer.add_scalar(f'Domain_Shift/PlantVillage_{metric}_Phone', data.get('phone', 0), 0)
                self.writer.add_scalar(f'Domain_Shift/PlantVillage_{metric}_Delta', data.get('delta', 0), 0)
        
        phone_adaptation = analysis['phone_domain_adaptation']
        for metric, data in phone_adaptation.items():
            if isinstance(data, dict):
                self.writer.add_scalar(f'Domain_Shift/Phone_{metric}_Improvement', data.get('improvement', 0), 0)
    
    def _save_failure_cases(self):
        """Save failure cases for analysis"""
        print("💾 Saving failure cases...")
        
        failure_dir = self.output_dir / 'failure_cases'
        failure_dir.mkdir(exist_ok=True)
        
        # Save PlantVillage model failures
        plantvillage_failures = self.results['plantvillage_model']
        if plantvillage_failures:
            plantvillage_dir = failure_dir / 'plantvillage_model'
            plantvillage_dir.mkdir(exist_ok=True)
            
            for dataset, failures in [('fgvc', plantvillage_failures.get('fgvc_failures', [])), 
                                    ('phone', plantvillage_failures.get('phone_failures', []))]:
                dataset_dir = plantvillage_dir / dataset
                dataset_dir.mkdir(exist_ok=True)
                
                for i, case in enumerate(failures[:50]):  # Save first 50 failures
                    # Save image
                    img = case['image'].permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = img.astype(np.uint8)
                    
                    img_path = dataset_dir / f'failure_{i:04d}.png'
                    Image.fromarray(img).save(img_path)
                    
                    # Save metadata
                    metadata = {
                        'true_label': case['true_label'],
                        'predicted_label': case['predicted_label'],
                        'confidence': case['confidence']
                    }
                    
                    metadata_path = dataset_dir / f'failure_{i:04d}.json'
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
        
        # Save phone model failures
        phone_failures = self.results['phone_model']
        if phone_failures:
            phone_dir = failure_dir / 'phone_model'
            phone_dir.mkdir(exist_ok=True)
            
            for dataset, failures in [('fgvc', phone_failures.get('fgvc_failures', [])), 
                                    ('phone', phone_failures.get('phone_failures', []))]:
                dataset_dir = phone_dir / dataset
                dataset_dir.mkdir(exist_ok=True)
                
                for i, case in enumerate(failures[:50]):  # Save first 50 failures
                    # Save image
                    img = case['image'].permute(1, 2, 0).numpy()
                    img = (img - img.min()) / (img.max() - img.min()) * 255
                    img = img.astype(np.uint8)
                    
                    img_path = dataset_dir / f'failure_{i:04d}.png'
                    Image.fromarray(img).save(img_path)
                    
                    # Save metadata
                    metadata = {
                        'true_label': case['true_label'],
                        'predicted_label': case['predicted_label'],
                        'confidence': case['confidence']
                    }
                    
                    metadata_path = dataset_dir / f'failure_{i:04d}.json'
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
        
        print(f"✅ Failure cases saved to {failure_dir}")
    
    def _generate_domain_shift_report(self):
        """Generate comprehensive domain shift report"""
        print("📋 Generating domain shift report...")
        
        analysis = self.results['domain_shift_analysis']
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'plantvillage_domain_shift': analysis['plantvillage_domain_shift'],
                'phone_domain_adaptation': analysis['phone_domain_adaptation'],
                'cross_domain_comparison': analysis['cross_domain_comparison'],
                'recommendations': analysis['recommendations']
            },
            'visualizations': [
                'domain_shift_comparison.png',
                'model_performance_heatmap.png',
                'delta_analysis.png',
                'cross_domain_comparison.png'
            ]
        }
        
        # Save JSON report
        report_path = self.output_dir / 'domain_shift_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        print(f"✅ Domain shift report saved to {report_path}")
        return report
    
    def _generate_markdown_report(self, report):
        """Generate markdown domain shift report"""
        md_content = f"""# Plant Stress Detection - Domain Shift Analysis Report

Generated: {report['timestamp']}

## Executive Summary

This report presents comprehensive domain shift analysis between PlantVillage and phone datasets, comparing the performance of PlantVillage-only and phone-adapted models.

## PlantVillage Model Domain Shift

| Metric | FGVC Performance | Phone Performance | Delta | % Change |
|--------|------------------|-------------------|-------|----------|
"""
        
        plantvillage_shift = report['summary']['plantvillage_domain_shift']
        for metric, data in plantvillage_shift.items():
            if isinstance(data, dict):
                md_content += f"| {metric} | {data.get('fgvc', 0):.4f} | {data.get('phone', 0):.4f} | {data.get('delta', 0):+.4f} | {data.get('percent_change', 0):+.1f}% |\n"
        
        md_content += "\n## Phone Model Domain Adaptation\n\n"
        md_content += "| Metric | PlantVillage Delta | Phone Delta | Improvement | % Improvement |\n"
        md_content += "|--------|-------------------|-------------|-------------|---------------|\n"
        
        phone_adaptation = report['summary']['phone_domain_adaptation']
        for metric, data in phone_adaptation.items():
            if isinstance(data, dict):
                md_content += f"| {metric} | {data.get('plantvillage_delta', 0):+.4f} | {data.get('phone_delta', 0):+.4f} | {data.get('improvement', 0):+.4f} | {data.get('improvement_percent', 0):+.1f}% |\n"
        
        md_content += "\n## Cross-Domain Model Comparison\n\n"
        
        cross_comparison = report['summary']['cross_domain_comparison']
        
        md_content += "### On FGVC Data\n\n"
        md_content += "| Metric | Difference (Phone - PlantVillage) |\n"
        md_content += "|--------|-----------------------------------|\n"
        
        fgvc_comparison = cross_comparison['plantvillage_vs_phone_on_fgvc']
        for metric, diff in fgvc_comparison.items():
            md_content += f"| {metric} | {diff:+.4f} |\n"
        
        md_content += "\n### On Phone Data\n\n"
        md_content += "| Metric | Difference (Phone - PlantVillage) |\n"
        md_content += "|--------|-----------------------------------|\n"
        
        phone_comparison = cross_comparison['plantvillage_vs_phone_on_phone']
        for metric, diff in phone_comparison.items():
            md_content += f"| {metric} | {diff:+.4f} |\n"
        
        md_content += "\n## Recommendations\n\n"
        
        for rec in report['summary']['recommendations']:
            md_content += f"- {rec}\n"
        
        md_content += "\n## Visualizations\n\n"
        
        for viz in report['visualizations']:
            md_content += f"- {viz}\n"
        
        # Save markdown report
        md_path = self.output_dir / 'domain_shift_report.md'
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        print(f"✅ Markdown report saved to {md_path}")
    
    def close(self):
        """Close TensorBoard writer and cleanup"""
        self.writer.close()
        print("✅ Domain shift evaluator closed")

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Domain Shift Evaluator")
    parser.add_argument("--plantvillage-model", required=True, help="Path to PlantVillage model")
    parser.add_argument("--phone-model", required=True, help="Path to phone model")
    parser.add_argument("--fgvc-data", required=True, help="Path to FGVC test data")
    parser.add_argument("--phone-data", required=True, help="Path to phone test data")
    parser.add_argument("--output-dir", default="domain_shift_results", help="Output directory")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 5")
    print("Domain Shift Evaluator")
    print("=" * 60)
    print(f"PlantVillage model: {args.plantvillage_model}")
    print(f"Phone model: {args.phone_model}")
    print(f"FGVC data: {args.fgvc_data}")
    print(f"Phone data: {args.phone_data}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Initialize evaluator
    evaluator = DomainShiftEvaluator(output_dir=args.output_dir)
    
    try:
        # Run domain shift evaluation
        analysis = evaluator.evaluate_domain_shift(
            args.plantvillage_model,
            args.phone_model,
            args.fgvc_data,
            args.phone_data
        )
        
        print("\n🎉 Domain shift evaluation completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"TensorBoard logs: {evaluator.tensorboard_dir}")
        
    finally:
        evaluator.close()

if __name__ == "__main__":
    main()
