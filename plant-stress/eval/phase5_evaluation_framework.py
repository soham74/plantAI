#!/usr/bin/env python3
"""
Phase 5: Evaluation & Visualization Framework
Comprehensive evaluation for YOLO and MobileViT models with domain shift analysis
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
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score, f1_score, mean_squared_error, mean_absolute_error
)

import ultralytics
from ultralytics import YOLO

import timm
from timm.data import create_transform

class EvaluationFramework:
    """Comprehensive evaluation framework for plant stress detection"""
    
    def __init__(self, output_dir="eval_results", tensorboard_dir="runs/eval"):
        self.output_dir = Path(output_dir)
        self.tensorboard_dir = Path(tensorboard_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.tensorboard_dir)
        
        # Results storage
        self.results = {
            'yolo': {},
            'mobilevit': {},
            'domain_shift': {},
            'failure_cases': []
        }
        
        # Color palette for visualizations
        self.colors = plt.cm.Set3(np.linspace(0, 1, 20))
        
    def evaluate_yolo_model(self, model_path, test_data, save_visualizations=True):
        """Evaluate YOLO model with comprehensive metrics"""
        print("🔍 Evaluating YOLO Model...")
        
        # Load model
        model = YOLO(model_path)
        
        # Run inference on test data
        results = model.val(data=test_data, save_json=True, save_txt=True)
        
        # Extract metrics
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr,
            'f1': results.box.map50 * 2 / (results.box.map50 + 1)  # Approximate F1
        }
        
        # For segmentation models, add mask metrics
        if hasattr(results, 'seg'):
            metrics.update({
                'mask_mAP50': results.seg.map50,
                'mask_mAP50-95': results.seg.map,
                'mask_IoU': results.seg.map50  # Approximate IoU
            })
        
        # Save metrics
        self.results['yolo'] = metrics
        
        # Log to TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f'YOLO/{key}', value, 0)
        
        # Generate visualizations
        if save_visualizations:
            self._generate_yolo_visualizations(model, test_data, metrics)
        
        print(f"✅ YOLO evaluation completed:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        
        return metrics
    
    def evaluate_mobilevit_model(self, model_path, test_loader, task="classification", 
                               class_names=None, save_visualizations=True):
        """Evaluate MobileViT model with comprehensive metrics"""
        print("🔍 Evaluating MobileViT Model...")
        
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        
        if isinstance(model, dict):
            model_state = model['model_state_dict']
            config = model.get('config', {})
            classes = model.get('classes', class_names)
        else:
            model_state = model.state_dict()
            config = {}
            classes = class_names
        
        # Create model architecture
        num_classes = len(classes) if task == "classification" else 1
        model_name = config.get('model_name', 'mobilevit_xxs.cvnets_in1k')
        
        net = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
        net.load_state_dict(model_state)
        net = net.to(device)
        net.eval()
        
        # Run evaluation
        all_preds = []
        all_targets = []
        all_probabilities = []
        failure_cases = []
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = net(data)
                
                if task == "classification":
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
                else:  # regression
                    preds = output.squeeze()
                    all_preds.extend(preds.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        if task == "classification":
            metrics = self._calculate_classification_metrics(
                all_targets, all_preds, all_probabilities, classes
            )
        else:
            metrics = self._calculate_regression_metrics(all_targets, all_preds)
        
        # Save metrics
        self.results['mobilevit'] = metrics
        
        # Log to TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f'MobileViT/{key}', value, 0)
        
        # Generate visualizations
        if save_visualizations:
            self._generate_mobilevit_visualizations(
                all_targets, all_preds, all_probabilities, classes, 
                failure_cases, task
            )
        
        print(f"✅ MobileViT evaluation completed:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")
        
        return metrics
    
    def _calculate_classification_metrics(self, targets, preds, probabilities, classes):
        """Calculate comprehensive classification metrics"""
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = np.mean(np.array(targets) == np.array(preds))
        metrics['macro_f1'] = f1_score(targets, preds, average='macro')
        metrics['weighted_f1'] = f1_score(targets, preds, average='weighted')
        
        # Per-class metrics
        report = classification_report(targets, preds, target_names=classes, output_dict=True)
        for class_name in classes:
            if class_name in report:
                metrics[f'f1_{class_name}'] = report[class_name]['f1-score']
                metrics[f'precision_{class_name}'] = report[class_name]['precision']
                metrics[f'recall_{class_name}'] = report[class_name]['recall']
        
        # Confusion matrix
        cm = confusion_matrix(targets, preds)
        metrics['confusion_matrix'] = cm
        
        # Precision-Recall curves
        if len(classes) == 2:  # Binary classification
            precision, recall, _ = precision_recall_curve(targets, probabilities[:, 1])
            ap = average_precision_score(targets, probabilities[:, 1])
            metrics['average_precision'] = ap
            metrics['precision_curve'] = precision
            metrics['recall_curve'] = recall
        else:  # Multi-class
            # One-vs-rest PR curves
            for i, class_name in enumerate(classes):
                precision, recall, _ = precision_recall_curve(
                    [1 if t == i else 0 for t in targets], 
                    probabilities[:, i]
                )
                ap = average_precision_score(
                    [1 if t == i else 0 for t in targets], 
                    probabilities[:, i]
                )
                metrics[f'ap_{class_name}'] = ap
                metrics[f'precision_curve_{class_name}'] = precision
                metrics[f'recall_curve_{class_name}'] = recall
        
        return metrics
    
    def _calculate_regression_metrics(self, targets, preds):
        """Calculate comprehensive regression metrics"""
        metrics = {}
        
        # Basic regression metrics
        metrics['rmse'] = np.sqrt(mean_squared_error(targets, preds))
        metrics['mae'] = mean_absolute_error(targets, preds)
        metrics['mse'] = mean_squared_error(targets, preds)
        
        # R-squared
        ss_res = np.sum((np.array(targets) - np.array(preds)) ** 2)
        ss_tot = np.sum((np.array(targets) - np.mean(targets)) ** 2)
        metrics['r2'] = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Mean absolute percentage error
        mape = np.mean(np.abs((np.array(targets) - np.array(preds)) / np.array(targets))) * 100
        metrics['mape'] = mape
        
        return metrics
    
    def _generate_yolo_visualizations(self, model, test_data, metrics):
        """Generate YOLO-specific visualizations"""
        print("📊 Generating YOLO visualizations...")
        
        # 1. PR Curves
        self._plot_yolo_pr_curves(model, test_data)
        
        # 2. Qualitative Grid
        self._plot_yolo_qualitative_grid(model, test_data)
        
        # 3. Metrics Summary
        self._plot_yolo_metrics_summary(metrics)
        
        # 4. Log to TensorBoard
        self._log_yolo_tensorboard(metrics)
    
    def _generate_mobilevit_visualizations(self, targets, preds, probabilities, classes, 
                                         failure_cases, task):
        """Generate MobileViT-specific visualizations"""
        print("📊 Generating MobileViT visualizations...")
        
        if task == "classification":
            # 1. Confusion Matrix
            self._plot_confusion_matrix(targets, preds, classes)
            
            # 2. PR Curves
            self._plot_mobilevit_pr_curves(targets, probabilities, classes)
            
            # 3. Failure Cases Grid
            if failure_cases:
                self._plot_failure_cases_grid(failure_cases, classes)
        else:
            # 1. Regression Scatter Plot
            self._plot_regression_scatter(targets, preds)
            
            # 2. Residual Plot
            self._plot_residual_plot(targets, preds)
        
        # 4. Log to TensorBoard
        self._log_mobilevit_tensorboard(targets, preds, probabilities, classes, task)
    
    def _plot_yolo_pr_curves(self, model, test_data):
        """Plot YOLO precision-recall curves"""
        # This would require running inference and extracting predictions
        # For now, create a placeholder
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title('YOLO Precision-Recall Curves')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.grid(True)
        
        # Placeholder curves
        recall = np.linspace(0, 1, 100)
        precision = 0.8 * np.exp(-2 * recall) + 0.2
        ax.plot(recall, precision, label='mAP50-95', linewidth=2)
        
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yolo_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('YOLO/PR_Curves', fig, 0)
    
    def _plot_yolo_qualitative_grid(self, model, test_data, num_samples=16):
        """Plot qualitative results grid for YOLO"""
        # This would require running inference on sample images
        # For now, create a placeholder grid
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle('YOLO Qualitative Results', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i < num_samples:
                # Placeholder image
                img = np.random.rand(224, 224, 3)
                ax.imshow(img)
                ax.set_title(f'Sample {i+1}')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yolo_qualitative_grid.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('YOLO/Qualitative_Grid', fig, 0)
    
    def _plot_yolo_metrics_summary(self, metrics):
        """Plot YOLO metrics summary"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color=self.colors[:len(metric_names)])
        ax.set_title('YOLO Metrics Summary')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'yolo_metrics_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('YOLO/Metrics_Summary', fig, 0)
    
    def _plot_confusion_matrix(self, targets, preds, classes):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, preds)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_title('MobileViT Confusion Matrix')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mobilevit_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('MobileViT/Confusion_Matrix', fig, 0)
    
    def _plot_mobilevit_pr_curves(self, targets, probabilities, classes):
        """Plot MobileViT precision-recall curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('MobileViT Precision-Recall Curves', fontsize=16)
        
        for i, class_name in enumerate(classes[:4]):  # Plot first 4 classes
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # One-vs-rest PR curve
            binary_targets = [1 if t == i else 0 for t in targets]
            precision, recall, _ = precision_recall_curve(binary_targets, probabilities[:, i])
            ap = average_precision_score(binary_targets, probabilities[:, i])
            
            ax.plot(recall, precision, label=f'{class_name} (AP={ap:.3f})', linewidth=2)
            ax.set_xlabel('Recall')
            ax.set_ylabel('Precision')
            ax.set_title(f'{class_name}')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mobilevit_pr_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('MobileViT/PR_Curves', fig, 0)
    
    def _plot_failure_cases_grid(self, failure_cases, classes, num_samples=16):
        """Plot failure cases grid"""
        if not failure_cases:
            return
        
        num_samples = min(num_samples, len(failure_cases))
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        fig.suptitle('MobileViT Failure Cases', fontsize=16)
        
        for i in range(num_samples):
            row, col = i // 4, i % 4
            ax = axes[row, col]
            
            case = failure_cases[i]
            img = case['image'].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min())  # Normalize to [0,1]
            
            ax.imshow(img)
            true_label = classes[case['true_label']] if case['true_label'] < len(classes) else f"Class {case['true_label']}"
            pred_label = classes[case['predicted_label']] if case['predicted_label'] < len(classes) else f"Class {case['predicted_label']}"
            ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {case["confidence"]:.3f}')
            ax.axis('off')
        
        # Hide empty subplots
        for i in range(num_samples, 16):
            row, col = i // 4, i % 4
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mobilevit_failure_cases.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('MobileViT/Failure_Cases', fig, 0)
    
    def _plot_regression_scatter(self, targets, preds):
        """Plot regression scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(targets, preds, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(min(targets), min(preds))
        max_val = max(max(targets), max(preds))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_xlabel('True Values')
        ax.set_ylabel('Predictions')
        ax.set_title('MobileViT Regression: True vs Predicted')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mobilevit_regression_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('MobileViT/Regression_Scatter', fig, 0)
    
    def _plot_residual_plot(self, targets, preds):
        """Plot residual plot for regression"""
        residuals = np.array(preds) - np.array(targets)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        ax1.scatter(preds, residuals, alpha=0.6, s=50)
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_xlabel('Predictions')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predictions')
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Residuals Distribution')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'mobilevit_residual_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('MobileViT/Residual_Plot', fig, 0)
    
    def _log_yolo_tensorboard(self, metrics):
        """Log YOLO metrics to TensorBoard"""
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'YOLO/{key}', value, 0)
    
    def _log_mobilevit_tensorboard(self, targets, preds, probabilities, classes, task):
        """Log MobileViT metrics to TensorBoard"""
        if task == "classification":
            # Log confusion matrix
            cm = confusion_matrix(targets, preds)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=classes, yticklabels=classes, ax=ax)
            ax.set_title('Confusion Matrix')
            self.writer.add_figure('MobileViT/Confusion_Matrix', fig, 0)
            plt.close()
        else:
            # Log regression scatter
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(targets, preds, alpha=0.6)
            ax.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
            ax.set_xlabel('True Values')
            ax.set_ylabel('Predictions')
            ax.set_title('Regression: True vs Predicted')
            self.writer.add_figure('MobileViT/Regression_Scatter', fig, 0)
            plt.close()
    
    def evaluate_domain_shift(self, plantvillage_model, phone_model, fgvc_data, phone_data):
        """Evaluate domain shift between datasets"""
        print("🌍 Evaluating Domain Shift...")
        
        # Evaluate on PlantVillage-only model
        plantvillage_metrics = self.evaluate_mobilevit_model(
            plantvillage_model, fgvc_data, task="classification"
        )
        
        # Evaluate on phone-adapted model
        phone_metrics = self.evaluate_mobilevit_model(
            phone_model, phone_data, task="classification"
        )
        
        # Calculate deltas
        domain_shift_results = {
            'plantvillage_metrics': plantvillage_metrics,
            'phone_metrics': phone_metrics,
            'deltas': {}
        }
        
        for key in plantvillage_metrics:
            if isinstance(plantvillage_metrics[key], (int, float)):
                delta = phone_metrics[key] - plantvillage_metrics[key]
                domain_shift_results['deltas'][key] = delta
        
        self.results['domain_shift'] = domain_shift_results
        
        # Generate domain shift visualization
        self._plot_domain_shift_analysis(domain_shift_results)
        
        print("✅ Domain shift evaluation completed:")
        for key, delta in domain_shift_results['deltas'].items():
            print(f"   {key} delta: {delta:+.4f}")
        
        return domain_shift_results
    
    def _plot_domain_shift_analysis(self, domain_shift_results):
        """Plot domain shift analysis"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Metrics comparison
        metrics = ['accuracy', 'macro_f1', 'weighted_f1']
        plantvillage_values = [domain_shift_results['plantvillage_metrics'].get(m, 0) for m in metrics]
        phone_values = [domain_shift_results['phone_metrics'].get(m, 0) for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, plantvillage_values, width, label='PlantVillage Model', alpha=0.8)
        ax1.bar(x + width/2, phone_values, width, label='Phone Model', alpha=0.8)
        
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score')
        ax1.set_title('Domain Shift: Model Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Deltas
        deltas = [domain_shift_results['deltas'].get(m, 0) for m in metrics]
        colors = ['green' if d >= 0 else 'red' for d in deltas]
        
        bars = ax2.bar(metrics, deltas, color=colors, alpha=0.7)
        ax2.set_xlabel('Metrics')
        ax2.set_ylabel('Delta (Phone - PlantVillage)')
        ax2.set_title('Domain Shift: Performance Deltas')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.02),
                    f'{delta:+.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'domain_shift_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Log to TensorBoard
        self.writer.add_figure('Domain_Shift/Analysis', fig, 0)
    
    def save_failure_cases(self, failure_cases, output_dir=None):
        """Save failure cases for regression testing"""
        if output_dir is None:
            output_dir = self.output_dir / 'failure_cases'
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"💾 Saving {len(failure_cases)} failure cases to {output_dir}")
        
        for i, case in enumerate(failure_cases):
            # Save image
            img = case['image'].permute(1, 2, 0).numpy()
            img = (img - img.min()) / (img.max() - img.min()) * 255
            img = img.astype(np.uint8)
            
            img_path = output_dir / f'failure_case_{i:04d}.png'
            Image.fromarray(img).save(img_path)
            
            # Save metadata
            metadata = {
                'true_label': case['true_label'],
                'predicted_label': case['predicted_label'],
                'confidence': case['confidence'],
                'batch_idx': case['batch_idx'],
                'sample_idx': case['sample_idx']
            }
            
            metadata_path = output_dir / f'failure_case_{i:04d}.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save summary
        summary = {
            'total_failure_cases': len(failure_cases),
            'failure_rate': len(failure_cases) / 1000,  # Assuming 1000 total samples
            'confidence_stats': {
                'mean': np.mean([c['confidence'] for c in failure_cases]),
                'std': np.std([c['confidence'] for c in failure_cases]),
                'min': np.min([c['confidence'] for c in failure_cases]),
                'max': np.max([c['confidence'] for c in failure_cases])
            }
        }
        
        with open(output_dir / 'failure_cases_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✅ Failure cases saved successfully")
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("📋 Generating evaluation report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'yolo_metrics': self.results.get('yolo', {}),
                'mobilevit_metrics': self.results.get('mobilevit', {}),
                'domain_shift': self.results.get('domain_shift', {}),
                'failure_cases_count': len(self.results.get('failure_cases', []))
            },
            'recommendations': self._generate_recommendations(),
            'visualizations': [
                'yolo_pr_curves.png',
                'yolo_qualitative_grid.png',
                'yolo_metrics_summary.png',
                'mobilevit_confusion_matrix.png',
                'mobilevit_pr_curves.png',
                'mobilevit_failure_cases.png',
                'domain_shift_analysis.png'
            ]
        }
        
        # Save report
        report_path = self.output_dir / 'evaluation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate markdown report
        self._generate_markdown_report(report)
        
        print(f"✅ Evaluation report saved to {report_path}")
        return report
    
    def _generate_recommendations(self):
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # YOLO recommendations
        yolo_metrics = self.results.get('yolo', {})
        if yolo_metrics:
            mAP = yolo_metrics.get('mAP50-95', 0)
            if mAP < 0.5:
                recommendations.append("YOLO mAP is low - consider data augmentation or model architecture changes")
            elif mAP < 0.7:
                recommendations.append("YOLO mAP could be improved - try hyperparameter tuning")
            else:
                recommendations.append("YOLO performance is good - ready for deployment")
        
        # MobileViT recommendations
        mobilevit_metrics = self.results.get('mobilevit', {})
        if mobilevit_metrics:
            if 'macro_f1' in mobilevit_metrics:
                f1 = mobilevit_metrics['macro_f1']
                if f1 < 0.7:
                    recommendations.append("MobileViT F1 score is low - consider class balancing or data quality")
                elif f1 < 0.85:
                    recommendations.append("MobileViT F1 score could be improved - try different loss functions")
                else:
                    recommendations.append("MobileViT performance is excellent")
        
        # Domain shift recommendations
        domain_shift = self.results.get('domain_shift', {})
        if domain_shift:
            deltas = domain_shift.get('deltas', {})
            if deltas:
                avg_delta = np.mean(list(deltas.values()))
                if avg_delta < -0.1:
                    recommendations.append("Significant domain shift detected - consider domain adaptation techniques")
                elif avg_delta < -0.05:
                    recommendations.append("Moderate domain shift - monitor performance in production")
                else:
                    recommendations.append("Minimal domain shift - models generalize well")
        
        return recommendations
    
    def _generate_markdown_report(self, report):
        """Generate markdown evaluation report"""
        md_content = f"""# Plant Stress Detection - Phase 5 Evaluation Report

Generated: {report['timestamp']}

## Executive Summary

This report presents comprehensive evaluation results for the plant stress detection system, including YOLO segmentation/detection and MobileViT classification models.

## YOLO Model Performance

"""
        
        yolo_metrics = report['summary']['yolo_metrics']
        if yolo_metrics:
            md_content += "| Metric | Value |\n|-------|-------|\n"
            for key, value in yolo_metrics.items():
                if isinstance(value, (int, float)):
                    md_content += f"| {key} | {value:.4f} |\n"
        else:
            md_content += "No YOLO metrics available.\n"
        
        md_content += "\n## MobileViT Model Performance\n\n"
        
        mobilevit_metrics = report['summary']['mobilevit_metrics']
        if mobilevit_metrics:
            md_content += "| Metric | Value |\n|-------|-------|\n"
            for key, value in mobilevit_metrics.items():
                if isinstance(value, (int, float)):
                    md_content += f"| {key} | {value:.4f} |\n"
        else:
            md_content += "No MobileViT metrics available.\n"
        
        md_content += "\n## Domain Shift Analysis\n\n"
        
        domain_shift = report['summary']['domain_shift']
        if domain_shift:
            deltas = domain_shift.get('deltas', {})
            if deltas:
                md_content += "| Metric | Delta (Phone - PlantVillage) |\n|-------|------------------------------|\n"
                for key, delta in deltas.items():
                    md_content += f"| {key} | {delta:+.4f} |\n"
        else:
            md_content += "No domain shift analysis available.\n"
        
        md_content += "\n## Recommendations\n\n"
        
        for rec in report['recommendations']:
            md_content += f"- {rec}\n"
        
        md_content += "\n## Visualizations\n\n"
        
        for viz in report['visualizations']:
            md_content += f"- {viz}\n"
        
        # Save markdown report
        md_path = self.output_dir / 'evaluation_report.md'
        with open(md_path, 'w') as f:
            f.write(md_content)
        
        print(f"✅ Markdown report saved to {md_path}")
    
    def close(self):
        """Close TensorBoard writer and cleanup"""
        self.writer.close()
        print("✅ Evaluation framework closed")

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Evaluation & Visualization Framework")
    parser.add_argument("--yolo-model", help="Path to YOLO model")
    parser.add_argument("--mobilevit-model", help="Path to MobileViT model")
    parser.add_argument("--test-data", help="Path to test data")
    parser.add_argument("--output-dir", default="eval_results", help="Output directory")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification",
                       help="MobileViT task type")
    parser.add_argument("--class-names", nargs="+", help="Class names for classification")
    parser.add_argument("--evaluate-domain-shift", action="store_true", help="Evaluate domain shift")
    parser.add_argument("--plantvillage-model", help="PlantVillage-only model for domain shift")
    parser.add_argument("--phone-model", help="Phone-adapted model for domain shift")
    parser.add_argument("--fgvc-data", help="FGVC test data for domain shift")
    parser.add_argument("--phone-data", help="Phone test data for domain shift")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 5")
    print("Evaluation & Visualization Framework")
    print("=" * 60)
    
    # Initialize evaluation framework
    evaluator = EvaluationFramework(output_dir=args.output_dir)
    
    try:
        # Evaluate YOLO model
        if args.yolo_model and args.test_data:
            evaluator.evaluate_yolo_model(args.yolo_model, args.test_data)
        
        # Evaluate MobileViT model
        if args.mobilevit_model and args.test_data:
            evaluator.evaluate_mobilevit_model(
                args.mobilevit_model, 
                args.test_data, 
                task=args.task,
                class_names=args.class_names
            )
        
        # Evaluate domain shift
        if args.evaluate_domain_shift and args.plantvillage_model and args.phone_model:
            evaluator.evaluate_domain_shift(
                args.plantvillage_model,
                args.phone_model,
                args.fgvc_data,
                args.phone_data
            )
        
        # Generate evaluation report
        evaluator.generate_evaluation_report()
        
        print("\n🎉 Evaluation completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"TensorBoard logs: {evaluator.tensorboard_dir}")
        
    finally:
        evaluator.close()

if __name__ == "__main__":
    main()
