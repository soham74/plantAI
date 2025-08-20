#!/usr/bin/env python3
"""
Phase 6: Performance Monitor
Real-time performance monitoring and optimization for plant stress detection
"""

import os
import sys
import time
import argparse
import json
import threading
from pathlib import Path
from collections import deque, defaultdict
from datetime import datetime

import cv2
import numpy as np
import torch
import psutil

class PerformanceMonitor:
    """Real-time performance monitoring system"""
    
    def __init__(self, log_interval=1.0):
        self.log_interval = log_interval
        self.running = False
        
        # Performance metrics
        self.fps_history = deque(maxlen=100)
        self.processing_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        self.gpu_usage = deque(maxlen=100)
        
        # Component timing
        self.component_times = defaultdict(lambda: deque(maxlen=100))
        
        # Performance alerts
        self.alerts = []
        
        # Monitoring thread
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start performance monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
        print("📊 Performance monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                # Collect system metrics
                self._collect_system_metrics()
                
                # Check for performance issues
                self._check_performance_alerts()
                
                # Log performance summary
                self._log_performance_summary()
                
                time.sleep(self.log_interval)
                
            except Exception as e:
                print(f"❌ Performance monitoring error: {e}")
    
    def _collect_system_metrics(self):
        """Collect system performance metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.cpu_usage.append(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.percent)
        
        # GPU usage (if available)
        try:
            if torch.cuda.is_available():
                gpu_percent = torch.cuda.utilization()
                self.gpu_usage.append(gpu_percent)
        except:
            self.gpu_usage.append(0)
    
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts"""
        current_time = datetime.now()
        
        # Check FPS
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            if avg_fps < 15:
                alert = {
                    'timestamp': current_time,
                    'type': 'low_fps',
                    'message': f'Low FPS detected: {avg_fps:.1f}',
                    'severity': 'warning'
                }
                self.alerts.append(alert)
        
        # Check processing time
        if self.processing_times:
            avg_processing = np.mean(self.processing_times)
            if avg_processing > 0.1:  # > 100ms
                alert = {
                    'timestamp': current_time,
                    'type': 'high_processing_time',
                    'message': f'High processing time: {avg_processing*1000:.1f}ms',
                    'severity': 'warning'
                }
                self.alerts.append(alert)
        
        # Check memory usage
        if self.memory_usage:
            avg_memory = np.mean(self.memory_usage)
            if avg_memory > 80:
                alert = {
                    'timestamp': current_time,
                    'type': 'high_memory_usage',
                    'message': f'High memory usage: {avg_memory:.1f}%',
                    'severity': 'critical'
                }
                self.alerts.append(alert)
        
        # Check CPU usage
        if self.cpu_usage:
            avg_cpu = np.mean(self.cpu_usage)
            if avg_cpu > 90:
                alert = {
                    'timestamp': current_time,
                    'type': 'high_cpu_usage',
                    'message': f'High CPU usage: {avg_cpu:.1f}%',
                    'severity': 'warning'
                }
                self.alerts.append(alert)
    
    def _log_performance_summary(self):
        """Log performance summary"""
        if not any([self.fps_history, self.processing_times, self.memory_usage, self.cpu_usage]):
            return
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'fps': {
                'current': self.fps_history[-1] if self.fps_history else 0,
                'average': np.mean(self.fps_history) if self.fps_history else 0,
                'min': np.min(self.fps_history) if self.fps_history else 0,
                'max': np.max(self.fps_history) if self.fps_history else 0
            },
            'processing_time': {
                'current': self.processing_times[-1] if self.processing_times else 0,
                'average': np.mean(self.processing_times) if self.processing_times else 0,
                'min': np.min(self.processing_times) if self.processing_times else 0,
                'max': np.max(self.processing_times) if self.processing_times else 0
            },
            'system': {
                'memory_percent': self.memory_usage[-1] if self.memory_usage else 0,
                'cpu_percent': self.cpu_usage[-1] if self.cpu_usage else 0,
                'gpu_percent': self.gpu_usage[-1] if self.gpu_usage else 0
            },
            'component_times': {
                component: {
                    'average': np.mean(times) if times else 0,
                    'max': np.max(times) if times else 0
                }
                for component, times in self.component_times.items()
            }
        }
        
        # Print summary to console
        print(f"\n📊 Performance Summary ({summary['timestamp']}):")
        print(f"  FPS: {summary['fps']['current']:.1f} (avg: {summary['fps']['average']:.1f})")
        print(f"  Processing: {summary['processing_time']['current']*1000:.1f}ms (avg: {summary['processing_time']['average']*1000:.1f}ms)")
        print(f"  Memory: {summary['system']['memory_percent']:.1f}%")
        print(f"  CPU: {summary['system']['cpu_percent']:.1f}%")
        if summary['system']['gpu_percent'] > 0:
            print(f"  GPU: {summary['system']['gpu_percent']:.1f}%")
    
    def record_fps(self, fps):
        """Record FPS measurement"""
        self.fps_history.append(fps)
    
    def record_processing_time(self, processing_time):
        """Record processing time measurement"""
        self.processing_times.append(processing_time)
    
    def record_component_time(self, component, time_taken):
        """Record component-specific timing"""
        self.component_times[component].append(time_taken)
    
    def get_performance_report(self):
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'fps': {
                    'average': np.mean(self.fps_history) if self.fps_history else 0,
                    'min': np.min(self.fps_history) if self.fps_history else 0,
                    'max': np.max(self.fps_history) if self.fps_history else 0,
                    'std': np.std(self.fps_history) if self.fps_history else 0
                },
                'processing_time': {
                    'average': np.mean(self.processing_times) if self.processing_times else 0,
                    'min': np.min(self.processing_times) if self.processing_times else 0,
                    'max': np.max(self.processing_times) if self.processing_times else 0,
                    'std': np.std(self.processing_times) if self.processing_times else 0
                },
                'system': {
                    'memory_average': np.mean(self.memory_usage) if self.memory_usage else 0,
                    'cpu_average': np.mean(self.cpu_usage) if self.cpu_usage else 0,
                    'gpu_average': np.mean(self.gpu_usage) if self.gpu_usage else 0
                }
            },
            'component_analysis': {
                component: {
                    'average': np.mean(times) if times else 0,
                    'max': np.max(times) if times else 0,
                    'percentage': np.mean(times) / np.mean(self.processing_times) * 100 if self.processing_times and times else 0
                }
                for component, times in self.component_times.items()
            },
            'alerts': self.alerts[-10:],  # Last 10 alerts
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self):
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # FPS recommendations
        if self.fps_history:
            avg_fps = np.mean(self.fps_history)
            if avg_fps < 20:
                recommendations.append({
                    'type': 'fps_optimization',
                    'priority': 'high',
                    'message': f'Low FPS ({avg_fps:.1f}). Consider:',
                    'suggestions': [
                        'Increase frame skip (--frame-skip 1 or 2)',
                        'Reduce camera resolution',
                        'Use smaller model variants',
                        'Enable GPU acceleration'
                    ]
                })
        
        # Processing time recommendations
        if self.processing_times:
            avg_processing = np.mean(self.processing_times)
            if avg_processing > 0.05:  # > 50ms
                recommendations.append({
                    'type': 'processing_optimization',
                    'priority': 'medium',
                    'message': f'High processing time ({avg_processing*1000:.1f}ms). Consider:',
                    'suggestions': [
                        'Optimize model inference',
                        'Reduce input resolution',
                        'Use model quantization',
                        'Enable batch processing'
                    ]
                })
        
        # Memory recommendations
        if self.memory_usage:
            avg_memory = np.mean(self.memory_usage)
            if avg_memory > 70:
                recommendations.append({
                    'type': 'memory_optimization',
                    'priority': 'high',
                    'message': f'High memory usage ({avg_memory:.1f}%). Consider:',
                    'suggestions': [
                        'Reduce batch size',
                        'Clear GPU cache periodically',
                        'Use memory-efficient models',
                        'Close unnecessary applications'
                    ]
                })
        
        # Component-specific recommendations
        if self.component_times:
            slowest_component = max(self.component_times.items(), 
                                  key=lambda x: np.mean(x[1]) if x[1] else 0)
            if slowest_component[1]:
                avg_time = np.mean(slowest_component[1])
                if avg_time > 0.02:  # > 20ms
                    recommendations.append({
                        'type': 'component_optimization',
                        'priority': 'medium',
                        'message': f'Slow component: {slowest_component[0]} ({avg_time*1000:.1f}ms). Consider:',
                        'suggestions': [
                            f'Optimize {slowest_component[0]} processing',
                            'Use hardware acceleration',
                            'Reduce input complexity'
                        ]
                    })
        
        return recommendations

class PerformanceOptimizer:
    """Performance optimization utilities"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_frame_skip(self, current_fps, target_fps=30):
        """Calculate optimal frame skip for target FPS"""
        if current_fps <= 0:
            return 0
        
        # Calculate how many frames to skip
        frame_skip = max(0, int(current_fps / target_fps) - 1)
        
        return min(frame_skip, 5)  # Cap at 5 frames
    
    def optimize_resolution(self, current_fps, current_resolution, target_fps=30):
        """Suggest optimal resolution for target FPS"""
        if current_fps >= target_fps:
            return current_resolution
        
        # Calculate scaling factor
        scale_factor = current_fps / target_fps
        
        # Suggest new resolution
        new_width = int(current_resolution[0] * scale_factor)
        new_height = int(current_resolution[1] * scale_factor)
        
        # Round to common resolutions
        resolutions = [(1920, 1080), (1280, 720), (854, 480), (640, 480)]
        
        for res in resolutions:
            if res[0] <= new_width and res[1] <= new_height:
                return res
        
        return (640, 480)  # Fallback
    
    def optimize_model_settings(self, performance_report):
        """Suggest model optimization settings"""
        recommendations = []
        
        # Check if GPU is being used effectively
        if performance_report['summary']['system']['gpu_average'] < 50:
            recommendations.append({
                'type': 'gpu_optimization',
                'setting': 'Enable GPU acceleration',
                'description': 'GPU usage is low, consider enabling CUDA/GPU acceleration'
            })
        
        # Check if model quantization would help
        if performance_report['summary']['processing_time']['average'] > 0.05:
            recommendations.append({
                'type': 'model_optimization',
                'setting': 'Model quantization',
                'description': 'Consider using quantized models for faster inference'
            })
        
        # Check if batch processing would help
        if performance_report['summary']['fps']['average'] < 20:
            recommendations.append({
                'type': 'batch_optimization',
                'setting': 'Batch processing',
                'description': 'Consider processing multiple frames in batches'
            })
        
        return recommendations

def main():
    parser = argparse.ArgumentParser(description="Phase 6: Performance Monitor")
    parser.add_argument("--log-interval", type=float, default=1.0,
                       help="Logging interval in seconds")
    parser.add_argument("--output-file", default="performance_report.json",
                       help="Output file for performance report")
    
    args = parser.parse_args()
    
    print("🌱 Plant Stress Detection - Phase 6")
    print("Performance Monitor")
    print("=" * 60)
    
    # Create performance monitor
    monitor = PerformanceMonitor(log_interval=args.log_interval)
    optimizer = PerformanceOptimizer()
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        print("📊 Performance monitoring active...")
        print("   Press Ctrl+C to stop and generate report")
        
        # Simulate some performance data (for testing)
        import random
        for i in range(10):
            monitor.record_fps(random.uniform(20, 30))
            monitor.record_processing_time(random.uniform(0.02, 0.05))
            monitor.record_component_time('yolo_inference', random.uniform(0.01, 0.03))
            monitor.record_component_time('mobilevit_inference', random.uniform(0.005, 0.015))
            time.sleep(1)
        
        # Keep running until interrupted
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Stopping performance monitoring...")
        
    finally:
        # Stop monitoring
        monitor.stop_monitoring()
        
        # Generate performance report
        report = monitor.get_performance_report()
        
        # Add optimization recommendations
        report['optimization_recommendations'] = optimizer.optimize_model_settings(report)
        
        # Save report
        with open(args.output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📊 Performance report saved to: {args.output_file}")
        
        # Print summary
        print("\n📋 Performance Summary:")
        print(f"  Average FPS: {report['summary']['fps']['average']:.1f}")
        print(f"  Average Processing Time: {report['summary']['processing_time']['average']*1000:.1f}ms")
        print(f"  Memory Usage: {report['summary']['system']['memory_average']:.1f}%")
        print(f"  CPU Usage: {report['summary']['system']['cpu_average']:.1f}%")
        
        if report['recommendations']:
            print("\n🔧 Recommendations:")
            for rec in report['recommendations']:
                print(f"  {rec['message']}")
                for suggestion in rec['suggestions']:
                    print(f"    - {suggestion}")

if __name__ == "__main__":
    main()
