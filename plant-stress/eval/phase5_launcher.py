#!/usr/bin/env python3
"""
Phase 5: Evaluation & Visualization Launcher
Complete evaluation pipeline for YOLO and MobileViT models with domain shift analysis
"""

import os
import sys
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

def run_yolo_evaluation(yolo_model_path, test_data_path, output_dir):
    """Run YOLO model evaluation"""
    print("🔍 Running YOLO Model Evaluation...")
    print("-" * 40)
    
    cmd = [
        sys.executable, "eval/phase5_evaluation_framework.py",
        "--yolo-model", yolo_model_path,
        "--test-data", test_data_path,
        "--output-dir", output_dir
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ YOLO evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ YOLO evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_mobilevit_evaluation(mobilevit_model_path, test_data_path, output_dir, 
                           task="classification", class_names=None):
    """Run MobileViT model evaluation"""
    print("🔍 Running MobileViT Model Evaluation...")
    print("-" * 40)
    
    cmd = [
        sys.executable, "eval/phase5_evaluation_framework.py",
        "--mobilevit-model", mobilevit_model_path,
        "--test-data", test_data_path,
        "--output-dir", output_dir,
        "--task", task
    ]
    
    if class_names:
        cmd.extend(["--class-names"] + class_names)
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ MobileViT evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ MobileViT evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def run_domain_shift_evaluation(plantvillage_model_path, phone_model_path, 
                              fgvc_data_path, phone_data_path, output_dir):
    """Run domain shift evaluation"""
    print("🌍 Running Domain Shift Evaluation...")
    print("-" * 40)
    
    cmd = [
        sys.executable, "eval/phase5_domain_shift_evaluator.py",
        "--plantvillage-model", plantvillage_model_path,
        "--phone-model", phone_model_path,
        "--fgvc-data", fgvc_data_path,
        "--phone-data", phone_data_path,
        "--output-dir", output_dir
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Domain shift evaluation completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Domain shift evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def create_evaluation_summary(output_dir, evaluations_run):
    """Create comprehensive evaluation summary"""
    print("📋 Creating evaluation summary...")
    
    summary = {
        'phase': 'Phase 5 - Evaluation & Visualization',
        'timestamp': datetime.now().isoformat(),
        'evaluations_run': evaluations_run,
        'output_directory': output_dir,
        'metrics_summary': {
            'yolo': {
                'mAP50-95': 'N/A',
                'mask_IoU': 'N/A',
                'precision': 'N/A',
                'recall': 'N/A'
            },
            'mobilevit': {
                'macro_f1': 'N/A',
                'accuracy': 'N/A',
                'confusion_matrix': 'Generated'
            },
            'domain_shift': {
                'plantvillage_delta': 'N/A',
                'phone_improvement': 'N/A',
                'cross_domain_comparison': 'Generated'
            }
        },
        'visualizations_generated': [
            'yolo_pr_curves.png',
            'yolo_qualitative_grid.png',
            'yolo_metrics_summary.png',
            'mobilevit_confusion_matrix.png',
            'mobilevit_pr_curves.png',
            'mobilevit_failure_cases.png',
            'domain_shift_comparison.png',
            'model_performance_heatmap.png',
            'delta_analysis.png',
            'cross_domain_comparison.png'
        ],
        'tensorboard_logs': f"{output_dir}/tensorboard",
        'failure_cases': f"{output_dir}/failure_cases"
    }
    
    # Save summary
    summary_path = Path(output_dir) / 'phase5_evaluation_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Generate markdown summary
    md_content = f"""# Plant Stress Detection - Phase 5 Evaluation Summary

Generated: {summary['timestamp']}

## Overview

This document summarizes the comprehensive evaluation and visualization results for the plant stress detection system, including YOLO segmentation/detection and MobileViT classification models.

## Evaluations Completed

"""
    
    for eval_type, status in evaluations_run.items():
        status_icon = "✅" if status else "❌"
        md_content += f"- {status_icon} {eval_type}: {'Completed' if status else 'Failed'}\n"
    
    md_content += f"""

## Output Directory

All results are saved to: `{output_dir}`

## Generated Visualizations

### YOLO Model
- **Precision-Recall Curves**: `yolo_pr_curves.png`
- **Qualitative Results Grid**: `yolo_qualitative_grid.png`
- **Metrics Summary**: `yolo_metrics_summary.png`

### MobileViT Model
- **Confusion Matrix**: `mobilevit_confusion_matrix.png`
- **Precision-Recall Curves**: `mobilevit_pr_curves.png`
- **Failure Cases Grid**: `mobilevit_failure_cases.png`

### Domain Shift Analysis
- **Model Comparison**: `domain_shift_comparison.png`
- **Performance Heatmap**: `model_performance_heatmap.png`
- **Delta Analysis**: `delta_analysis.png`
- **Cross-Domain Comparison**: `cross_domain_comparison.png`

## TensorBoard Logs

TensorBoard logs are available at: `{summary['tensorboard_logs']}`

To view the logs:
```bash
tensorboard --logdir {summary['tensorboard_logs']}
```

## Failure Cases

Failure cases have been saved for regression testing at: `{summary['failure_cases']}`

## Key Metrics

### YOLO (Detection/Segmentation)
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Mask IoU**: Intersection over Union for segmentation masks
- **Precision**: Precision at optimal threshold
- **Recall**: Recall at optimal threshold

### MobileViT (Classification)
- **Macro-F1**: Macro-averaged F1 score across all classes
- **Accuracy**: Overall classification accuracy
- **Confusion Matrix**: Per-class performance breakdown

### Domain Shift Analysis
- **PlantVillage Delta**: Performance change from FGVC to phone data
- **Phone Model Improvement**: Improvement over PlantVillage model
- **Cross-Domain Comparison**: Model comparison across datasets

## Recommendations

Based on the evaluation results:

1. **Model Performance**: Review individual model metrics for optimization opportunities
2. **Domain Shift**: Analyze domain shift results to determine if additional adaptation is needed
3. **Failure Cases**: Examine failure cases to identify common error patterns
4. **Visualizations**: Use generated visualizations for model comparison and debugging

## Next Steps

1. **Model Optimization**: Use evaluation results to guide hyperparameter tuning
2. **Data Quality**: Address issues identified in failure case analysis
3. **Domain Adaptation**: Implement additional domain adaptation if significant shift is detected
4. **Production Deployment**: Prepare models for real-time inference based on evaluation results

## Files Generated

- `evaluation_report.json`: Detailed evaluation metrics
- `evaluation_report.md`: Human-readable evaluation report
- `domain_shift_report.json`: Domain shift analysis results
- `domain_shift_report.md`: Domain shift analysis report
- `phase5_evaluation_summary.json`: This summary file
- `phase5_evaluation_summary.md`: This summary in markdown format

All visualizations and TensorBoard logs are saved in the output directory for further analysis.
"""
    
    # Save markdown summary
    md_path = Path(output_dir) / 'phase5_evaluation_summary.md'
    with open(md_path, 'w') as f:
        f.write(md_content)
    
    print(f"✅ Evaluation summary saved to {summary_path}")
    print(f"✅ Markdown summary saved to {md_path}")
    
    return summary

def main():
    parser = argparse.ArgumentParser(description="Phase 5: Evaluation & Visualization Launcher")
    parser.add_argument("--yolo-model", help="Path to YOLO model")
    parser.add_argument("--mobilevit-model", help="Path to MobileViT model")
    parser.add_argument("--test-data", help="Path to test data")
    parser.add_argument("--output-dir", default="phase5_evaluation_results", help="Output directory")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification",
                       help="MobileViT task type")
    parser.add_argument("--class-names", nargs="+", help="Class names for classification")
    parser.add_argument("--evaluate-domain-shift", action="store_true", help="Evaluate domain shift")
    parser.add_argument("--plantvillage-model", help="PlantVillage-only model for domain shift")
    parser.add_argument("--phone-model", help="Phone-adapted model for domain shift")
    parser.add_argument("--fgvc-data", help="FGVC test data for domain shift")
    parser.add_argument("--phone-data", help="Phone test data for domain shift")
    parser.add_argument("--skip-yolo", action="store_true", help="Skip YOLO evaluation")
    parser.add_argument("--skip-mobilevit", action="store_true", help="Skip MobileViT evaluation")
    parser.add_argument("--skip-domain-shift", action="store_true", help="Skip domain shift evaluation")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 5")
    print("Evaluation & Visualization Launcher")
    print("=" * 60)
    print(f"Output directory: {args.output_dir}")
    print(f"Task: {args.task}")
    print(f"Evaluate domain shift: {args.evaluate_domain_shift}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    evaluations_run = {}
    
    # Run YOLO evaluation
    if not args.skip_yolo and args.yolo_model and args.test_data:
        evaluations_run['YOLO Evaluation'] = run_yolo_evaluation(
            args.yolo_model, args.test_data, args.output_dir
        )
    else:
        evaluations_run['YOLO Evaluation'] = False
        if not args.skip_yolo:
            print("⚠️ Skipping YOLO evaluation - missing model or test data")
    
    # Run MobileViT evaluation
    if not args.skip_mobilevit and args.mobilevit_model and args.test_data:
        evaluations_run['MobileViT Evaluation'] = run_mobilevit_evaluation(
            args.mobilevit_model, args.test_data, args.output_dir, 
            args.task, args.class_names
        )
    else:
        evaluations_run['MobileViT Evaluation'] = False
        if not args.skip_mobilevit:
            print("⚠️ Skipping MobileViT evaluation - missing model or test data")
    
    # Run domain shift evaluation
    if (not args.skip_domain_shift and args.evaluate_domain_shift and 
        args.plantvillage_model and args.phone_model and args.fgvc_data and args.phone_data):
        evaluations_run['Domain Shift Evaluation'] = run_domain_shift_evaluation(
            args.plantvillage_model, args.phone_model, 
            args.fgvc_data, args.phone_data, args.output_dir
        )
    else:
        evaluations_run['Domain Shift Evaluation'] = False
        if not args.skip_domain_shift and args.evaluate_domain_shift:
            print("⚠️ Skipping domain shift evaluation - missing models or data")
    
    # Create evaluation summary
    summary = create_evaluation_summary(args.output_dir, evaluations_run)
    
    # Print final summary
    print("\n🎉 Phase 5 Evaluation Summary")
    print("=" * 40)
    
    successful_evaluations = sum(evaluations_run.values())
    total_evaluations = len(evaluations_run)
    
    print(f"Evaluations completed: {successful_evaluations}/{total_evaluations}")
    
    for eval_type, status in evaluations_run.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {eval_type}")
    
    print(f"\nResults saved to: {args.output_dir}")
    print(f"TensorBoard logs: {output_dir}/tensorboard")
    print(f"Failure cases: {output_dir}/failure_cases")
    
    if successful_evaluations > 0:
        print(f"\n📊 View results with:")
        print(f"  tensorboard --logdir {output_dir}/tensorboard")
        print(f"  open {output_dir}/phase5_evaluation_summary.md")
    
    print(f"\n📋 Generated files:")
    for viz in summary['visualizations_generated']:
        print(f"  - {viz}")
    
    print(f"\n📄 Reports:")
    print(f"  - evaluation_report.json")
    print(f"  - evaluation_report.md")
    if evaluations_run.get('Domain Shift Evaluation', False):
        print(f"  - domain_shift_report.json")
        print(f"  - domain_shift_report.md")
    print(f"  - phase5_evaluation_summary.json")
    print(f"  - phase5_evaluation_summary.md")

if __name__ == "__main__":
    main()
