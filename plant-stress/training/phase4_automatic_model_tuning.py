#!/usr/bin/env python3
"""
Phase 4: Automatic Model Tuning (AMT)
AWS SageMaker Automatic Model Tuning for hyperparameter optimization
"""

import os
import sys
import argparse
import json
import boto3
from pathlib import Path
from datetime import datetime
from botocore.exceptions import ClientError

def create_sagemaker_client(region_name="us-east-1"):
    """Create SageMaker client"""
    try:
        sagemaker_client = boto3.client('sagemaker', region_name=region_name)
        print(f"✅ SageMaker client created for region: {region_name}")
        return sagemaker_client
    except Exception as e:
        print(f"❌ Failed to create SageMaker client: {e}")
        return None

def create_hyperparameter_ranges(model_type="yolo"):
    """Create hyperparameter ranges for AMT"""
    
    if model_type == "yolo":
        # YOLO hyperparameter ranges
        hyperparameter_ranges = {
            'model_size': {
                'type': 'Categorical',
                'values': ['n', 's', 'm']
            },
            'img_size': {
                'type': 'Categorical',
                'values': [512, 640]
            },
            'epochs': {
                'type': 'Integer',
                'min_value': 50,
                'max_value': 200
            },
            'batch_size': {
                'type': 'Categorical',
                'values': [8, 16, 32]
            },
            'lr0': {
                'type': 'Continuous',
                'min_value': 0.001,
                'max_value': 0.1
            },
            'weight_decay': {
                'type': 'Continuous',
                'min_value': 0.0001,
                'max_value': 0.001
            },
            'hsv_h': {
                'type': 'Continuous',
                'min_value': 0.01,
                'max_value': 0.02
            },
            'hsv_s': {
                'type': 'Continuous',
                'min_value': 0.5,
                'max_value': 0.9
            },
            'hsv_v': {
                'type': 'Continuous',
                'min_value': 0.3,
                'max_value': 0.5
            },
            'mosaic': {
                'type': 'Categorical',
                'values': [0.0, 0.5, 1.0]
            },
            'fliplr': {
                'type': 'Categorical',
                'values': [0.0, 0.3, 0.5, 0.7]
            },
            'blur': {
                'type': 'Continuous',
                'min_value': 0.0,
                'max_value': 0.02
            }
        }
        
        # Objective metric for YOLO (detection/segmentation)
        objective_metric_name = "mAP50-95"
        objective_type = "Maximize"
        
    else:  # mobilevit
        # MobileViT hyperparameter ranges
        hyperparameter_ranges = {
            'model_name': {
                'type': 'Categorical',
                'values': [
                    'mobilevit_xxs.cvnets_in1k',
                    'mobilevit_xs.cvnets_in1k',
                    'mobilevit_s.cvnets_in1k'
                ]
            },
            'img_size': {
                'type': 'Categorical',
                'values': [224, 256]
            },
            'batch_size': {
                'type': 'Categorical',
                'values': [16, 32, 64]
            },
            'epochs': {
                'type': 'Integer',
                'min_value': 50,
                'max_value': 150
            },
            'lr': {
                'type': 'Continuous',
                'min_value': 1e-5,
                'max_value': 1e-3
            },
            'weight_decay': {
                'type': 'Continuous',
                'min_value': 1e-5,
                'max_value': 1e-3
            },
            'task': {
                'type': 'Categorical',
                'values': ['classification', 'regression']
            },
            'loss': {
                'type': 'Categorical',
                'values': ['ce', 'focal', 'smooth']
            },
            'use_balanced_sampler': {
                'type': 'Categorical',
                'values': [True, False]
            },
            'use_ema': {
                'type': 'Categorical',
                'values': [True, False]
            }
        }
        
        # Objective metric for MobileViT (classification/regression)
        objective_metric_name = "macro_f1" if model_type == "classification" else "rmse"
        objective_type = "Maximize" if model_type == "classification" else "Minimize"
    
    return hyperparameter_ranges, objective_metric_name, objective_type

def create_training_job_definition(
    job_name, 
    role_arn, 
    image_uri, 
    instance_type, 
    instance_count,
    hyperparameter_ranges,
    input_data_config,
    output_data_config,
    model_type="yolo"
):
    """Create training job definition for AMT"""
    
    # Static hyperparameters
    static_hyperparameters = {
        'sagemaker_program': f'phase4_sagemaker_{model_type}.py',
        'sagemaker_submit_directory': '/opt/ml/code',
        'sagemaker_region': 'us-east-1'
    }
    
    # Create training job definition
    training_job_definition = {
        'TrainingJobName': job_name,
        'RoleArn': role_arn,
        'AlgorithmSpecification': {
            'TrainingImage': image_uri,
            'TrainingInputMode': 'File'
        },
        'ResourceConfig': {
            'InstanceType': instance_type,
            'InstanceCount': instance_count,
            'VolumeSizeInGB': 100
        },
        'HyperParameterRanges': hyperparameter_ranges,
        'StaticHyperParameters': static_hyperparameters,
        'InputDataConfig': input_data_config,
        'OutputDataConfig': output_data_config,
        'StoppingCondition': {
            'MaxRuntimeInSeconds': 3600 * 24  # 24 hours
        }
    }
    
    return training_job_definition

def create_hyperparameter_tuning_job(
    sagemaker_client,
    tuning_job_name,
    training_job_definition,
    objective_metric_name,
    objective_type,
    max_jobs=20,
    max_parallel_jobs=4,
    strategy="Bayesian"
):
    """Create hyperparameter tuning job"""
    
    try:
        response = sagemaker_client.create_hyperparameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name,
            HyperParameterTuningJobConfig={
                'Strategy': strategy,
                'HyperParameterTuningJobObjective': {
                    'Type': objective_type,
                    'MetricName': objective_metric_name
                },
                'ResourceLimits': {
                    'MaxNumberOfTrainingJobs': max_jobs,
                    'MaxParallelTrainingJobs': max_parallel_jobs
                },
                'ParameterRanges': training_job_definition['HyperParameterRanges']
            },
            TrainingJobDefinition={
                'StaticHyperParameters': training_job_definition['StaticHyperParameters'],
                'AlgorithmSpecification': training_job_definition['AlgorithmSpecification'],
                'RoleArn': training_job_definition['RoleArn'],
                'InputDataConfig': training_job_definition['InputDataConfig'],
                'OutputDataConfig': training_job_definition['OutputDataConfig'],
                'ResourceConfig': training_job_definition['ResourceConfig'],
                'StoppingCondition': training_job_definition['StoppingCondition']
            }
        )
        
        print(f"✅ Hyperparameter tuning job created: {tuning_job_name}")
        print(f"   Job ARN: {response['HyperParameterTuningJobArn']}")
        return response['HyperParameterTuningJobArn']
        
    except ClientError as e:
        print(f"❌ Failed to create hyperparameter tuning job: {e}")
        return None

def monitor_tuning_job(sagemaker_client, tuning_job_name):
    """Monitor hyperparameter tuning job progress"""
    
    print(f"🔍 Monitoring tuning job: {tuning_job_name}")
    
    try:
        while True:
            response = sagemaker_client.describe_hyperparameter_tuning_job(
                HyperParameterTuningJobName=tuning_job_name
            )
            
            status = response['HyperParameterTuningJobStatus']
            print(f"Status: {status}")
            
            if 'HyperParameterTuningJobConfig' in response:
                config = response['HyperParameterTuningJobConfig']
                print(f"Strategy: {config.get('Strategy', 'N/A')}")
                print(f"Max jobs: {config['ResourceLimits']['MaxNumberOfTrainingJobs']}")
                print(f"Max parallel jobs: {config['ResourceLimits']['MaxParallelTrainingJobs']}")
            
            if 'TrainingJobStatusCounters' in response:
                counters = response['TrainingJobStatusCounters']
                print(f"Training jobs - Completed: {counters.get('Completed', 0)}, "
                      f"InProgress: {counters.get('InProgress', 0)}, "
                      f"Failed: {counters.get('Failed', 0)}")
            
            if status in ['Completed', 'Failed', 'Stopped']:
                print(f"🎉 Tuning job {status.lower()}")
                break
            
            import time
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        print("\n⏹️ Monitoring stopped by user")
    except ClientError as e:
        print(f"❌ Error monitoring tuning job: {e}")

def get_best_training_job(sagemaker_client, tuning_job_name):
    """Get the best training job from tuning results"""
    
    try:
        response = sagemaker_client.describe_hyperparameter_tuning_job(
            HyperParameterTuningJobName=tuning_job_name
        )
        
        if 'BestTrainingJob' in response:
            best_job = response['BestTrainingJob']
            print(f"🏆 Best training job: {best_job['TrainingJobName']}")
            print(f"   Best objective value: {best_job.get('FinalHyperParameterTuningJobObjectiveMetric', {}).get('Value', 'N/A')}")
            print(f"   Hyperparameters: {best_job.get('TunedHyperParameters', {})}")
            return best_job
        else:
            print("❌ No best training job found")
            return None
            
    except ClientError as e:
        print(f"❌ Error getting best training job: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Phase 4: Automatic Model Tuning")
    parser.add_argument("--model-type", choices=["yolo", "mobilevit"], default="yolo",
                       help="Model type for tuning")
    parser.add_argument("--role-arn", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--image-uri", required=True, help="Training container image URI")
    parser.add_argument("--instance-type", default="ml.p3.2xlarge", help="Training instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Number of instances")
    parser.add_argument("--s3-data-uri", required=True, help="S3 URI for training data")
    parser.add_argument("--s3-output-uri", required=True, help="S3 URI for model output")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--max-jobs", type=int, default=20, help="Maximum number of training jobs")
    parser.add_argument("--max-parallel-jobs", type=int, default=4, help="Maximum parallel jobs")
    parser.add_argument("--strategy", choices=["Bayesian", "Random"], default="Bayesian",
                       help="Tuning strategy")
    parser.add_argument("--action", choices=["create", "monitor", "get-best"], default="create",
                       help="Action to perform")
    parser.add_argument("--tuning-job-name", help="Tuning job name for monitor/get-best")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 4")
    print("Automatic Model Tuning (AMT)")
    print("=" * 60)
    print(f"Model type: {args.model_type}")
    print(f"Strategy: {args.strategy}")
    print(f"Max jobs: {args.max_jobs}")
    print(f"Max parallel jobs: {args.max_parallel_jobs}")
    print()
    
    # Create SageMaker client
    sagemaker_client = create_sagemaker_client(args.region)
    if not sagemaker_client:
        return
    
    if args.action == "create":
        # Create hyperparameter ranges
        hyperparameter_ranges, objective_metric_name, objective_type = create_hyperparameter_ranges(args.model_type)
        
        print(f"🎯 Objective metric: {objective_metric_name} ({objective_type})")
        print(f"📊 Hyperparameter ranges:")
        for param, config in hyperparameter_ranges.items():
            print(f"   {param}: {config}")
        print()
        
        # Create input/output data config
        input_data_config = [
            {
                'ChannelName': 'training',
                'DataSource': {
                    'S3DataSource': {
                        'S3DataType': 'S3Prefix',
                        'S3Uri': args.s3_data_uri,
                        'S3DataDistributionType': 'FullyReplicated'
                    }
                },
                'ContentType': 'application/zip'
            }
        ]
        
        output_data_config = {
            'S3OutputPath': args.s3_output_uri
        }
        
        # Create training job definition
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"plant-stress-{args.model_type}-training-{timestamp}"
        
        training_job_definition = create_training_job_definition(
            job_name,
            args.role_arn,
            args.image_uri,
            args.instance_type,
            args.instance_count,
            hyperparameter_ranges,
            input_data_config,
            output_data_config,
            args.model_type
        )
        
        # Create hyperparameter tuning job
        tuning_job_name = f"plant-stress-{args.model_type}-tuning-{timestamp}"
        
        tuning_job_arn = create_hyperparameter_tuning_job(
            sagemaker_client,
            tuning_job_name,
            training_job_definition,
            objective_metric_name,
            objective_type,
            args.max_jobs,
            args.max_parallel_jobs,
            args.strategy
        )
        
        if tuning_job_arn:
            print(f"\n🎉 Hyperparameter tuning job created successfully!")
            print(f"Tuning job name: {tuning_job_name}")
            print(f"Tuning job ARN: {tuning_job_arn}")
            print(f"\nMonitor progress with:")
            print(f"python {sys.argv[0]} --action monitor --tuning-job-name {tuning_job_name}")
    
    elif args.action == "monitor":
        if not args.tuning_job_name:
            print("❌ Tuning job name required for monitoring")
            return
        
        monitor_tuning_job(sagemaker_client, args.tuning_job_name)
    
    elif args.action == "get-best":
        if not args.tuning_job_name:
            print("❌ Tuning job name required for getting best job")
            return
        
        best_job = get_best_training_job(sagemaker_client, args.tuning_job_name)
        if best_job:
            print(f"\n✅ Best training job retrieved successfully!")

if __name__ == "__main__":
    main()
