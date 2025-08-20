#!/usr/bin/env python3
"""
Phase 6: Device Enumeration Utility
Specialized utility for enumerating and testing iPhone Continuity Camera devices
"""

import os
import sys
import subprocess
import cv2
import numpy as np
from pathlib import Path

def check_continuity_camera_prerequisites():
    """Check Continuity Camera prerequisites"""
    print("🔍 Checking Continuity Camera prerequisites...")
    print("=" * 50)
    
    prerequisites = [
        "✅ macOS 12.3 or later",
        "✅ iPhone with iOS 15.4 or later",
        "✅ Both devices signed in with same Apple ID",
        "✅ Wi-Fi and Bluetooth enabled on both devices",
        "✅ iPhone nearby and unlocked",
        "✅ Continuity Camera enabled in System Preferences"
    ]
    
    for prereq in prerequisites:
        print(f"  {prereq}")
    
    print("\n📱 To enable Continuity Camera:")
    print("  1. On Mac: System Preferences > General > AirPlay & Handoff")
    print("  2. Enable 'Allow Handoff between this Mac and your iCloud devices'")
    print("  3. On iPhone: Settings > General > AirPlay & Handoff")
    print("  4. Enable 'Handoff'")
    print("  5. In supported apps, look for the camera icon in the menu bar")

def enumerate_ffmpeg_devices():
    """Enumerate devices using FFmpeg"""
    print("\n🔍 Enumerating devices with FFmpeg...")
    print("-" * 30)
    
    try:
        # Run FFmpeg to list devices
        result = subprocess.run([
            'ffmpeg', '-f', 'avfoundation', '-list_devices', 'true', '-i', ''
        ], capture_output=True, text=True, timeout=15)
        
        if result.returncode == 0:
            print("📱 Available AVFoundation devices:")
            lines = result.stderr.split('\n')
            
            video_devices = []
            audio_devices = []
            
            for line in lines:
                line = line.strip()
                if 'AVFoundation video devices' in line:
                    print(f"\n🎥 {line}")
                elif 'AVFoundation audio devices' in line:
                    print(f"\n🎵 {line}")
                elif '[' in line and ']' in line and ('iPhone' in line or 'Continuity' in line or 'Camera' in line):
                    print(f"  📱 {line}")
                    # Extract device index
                    import re
                    match = re.search(r'\[(\d+)\]', line)
                    if match:
                        device_idx = int(match.group(1))
                        if 'video' in line.lower() or 'camera' in line.lower():
                            video_devices.append(device_idx)
                        elif 'audio' in line.lower():
                            audio_devices.append(device_idx)
            
            print(f"\n📊 Summary:")
            print(f"  Video devices found: {len(video_devices)}")
            print(f"  Audio devices found: {len(audio_devices)}")
            
            return video_devices, audio_devices
            
        else:
            print(f"❌ FFmpeg command failed: {result.stderr}")
            return [], []
            
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg command timed out")
        return [], []
    except FileNotFoundError:
        print("❌ FFmpeg not found. Install with: brew install ffmpeg")
        return [], []
    except Exception as e:
        print(f"❌ Error running FFmpeg: {e}")
        return [], []

def enumerate_opencv_devices():
    """Enumerate devices using OpenCV"""
    print("\n🔍 Enumerating devices with OpenCV...")
    print("-" * 30)
    
    available_devices = []
    
    # Try different backends
    backends = [
        (cv2.CAP_AVFOUNDATION, "AVFoundation"),
        (cv2.CAP_ANY, "Auto"),
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_DSHOW, "DirectShow")
    ]
    
    for backend, backend_name in backends:
        print(f"\n🎥 Testing {backend_name} backend:")
        
        for i in range(10):  # Check first 10 device indices
            try:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        print(f"  ✅ Device {i}: {width}x{height} @ {fps:.1f}fps")
                        available_devices.append({
                            'index': i,
                            'backend': backend,
                            'backend_name': backend_name,
                            'resolution': (width, height),
                            'fps': fps
                        })
                    else:
                        print(f"  ⚠️ Device {i}: Opened but no frame")
                    cap.release()
                else:
                    print(f"  ❌ Device {i}: Not available")
            except Exception as e:
                print(f"  ❌ Device {i}: Error - {e}")
    
    return available_devices

def test_device(device_info):
    """Test a specific device"""
    print(f"\n🧪 Testing device {device_info['index']} ({device_info['backend_name']})...")
    
    cap = cv2.VideoCapture(device_info['index'], device_info['backend'])
    
    if not cap.isOpened():
        print(f"❌ Failed to open device {device_info['index']}")
        return False
    
    # Set properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Read a few frames
    frames_read = 0
    for i in range(10):
        ret, frame = cap.read()
        if ret and frame is not None:
            frames_read += 1
            if i == 0:  # First frame
                actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                print(f"  📐 Resolution: {actual_width}x{actual_height}")
                print(f"  🎬 FPS: {actual_fps:.1f}")
                print(f"  📊 Frame size: {frame.shape}")
    
    cap.release()
    
    print(f"  ✅ Frames read: {frames_read}/10")
    return frames_read > 0

def find_iphone_device(available_devices):
    """Find iPhone/Continuity Camera device"""
    print("\n🎯 Looking for iPhone/Continuity Camera...")
    print("-" * 40)
    
    iphone_candidates = []
    
    for device in available_devices:
        # Test the device
        if test_device(device):
            # Heuristic: iPhone is usually not device 0 and has good resolution
            if device['index'] > 0 and device['resolution'][0] >= 1280:
                iphone_candidates.append(device)
                print(f"  🎯 Device {device['index']} is a good candidate for iPhone")
    
    if iphone_candidates:
        # Sort by resolution (higher is better)
        iphone_candidates.sort(key=lambda x: x['resolution'][0] * x['resolution'][1], reverse=True)
        best_device = iphone_candidates[0]
        print(f"\n🏆 Best iPhone candidate: Device {best_device['index']}")
        print(f"   Backend: {best_device['backend_name']}")
        print(f"   Resolution: {best_device['resolution'][0]}x{best_device['resolution'][1]}")
        print(f"   FPS: {best_device['fps']:.1f}")
        return best_device
    else:
        print("❌ No suitable iPhone device found")
        return None

def generate_device_config(device_info):
    """Generate device configuration"""
    if device_info is None:
        return None
    
    config = {
        'device_index': device_info['index'],
        'backend': device_info['backend_name'],
        'resolution': device_info['resolution'],
        'fps': device_info['fps'],
        'command_line': f"--device-index {device_info['index']} --resolution {device_info['resolution'][0]} {device_info['resolution'][1]}"
    }
    
    return config

def save_device_config(config, filename="continuity_camera_config.json"):
    """Save device configuration to file"""
    if config is None:
        return
    
    import json
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n💾 Device configuration saved to: {filename}")
    print(f"   Use command: python runtime/phase6_realtime_inference.py {config['command_line']}")

def main():
    print("🌱 Plant Stress Detection - Phase 6")
    print("Device Enumeration Utility")
    print("=" * 60)
    
    # Check prerequisites
    check_continuity_camera_prerequisites()
    
    # Enumerate devices with FFmpeg
    ffmpeg_video, ffmpeg_audio = enumerate_ffmpeg_devices()
    
    # Enumerate devices with OpenCV
    opencv_devices = enumerate_opencv_devices()
    
    if not opencv_devices:
        print("\n❌ No video devices found!")
        print("   Please check Continuity Camera setup and try again")
        return
    
    # Find iPhone device
    iphone_device = find_iphone_device(opencv_devices)
    
    # Generate and save configuration
    config = generate_device_config(iphone_device)
    save_device_config(config)
    
    print("\n🎉 Device enumeration complete!")
    print("\n📋 Next steps:")
    print("   1. Ensure iPhone is nearby and unlocked")
    print("   2. Run: python runtime/phase6_realtime_inference.py --list-devices")
    print("   3. Run: python runtime/phase6_realtime_inference.py [config options]")
    
    if iphone_device:
        print(f"\n🚀 Quick start:")
        print(f"   python runtime/phase6_realtime_inference.py {config['command_line']}")

if __name__ == "__main__":
    main()
