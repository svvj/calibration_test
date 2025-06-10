#!/usr/bin/env python3
import cv2
import os
import time

def list_available_cameras(max_cameras=10):
    """Check for available cameras by index."""
    available_cameras = []
    
    print("Checking camera indices:")
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  Camera index {i}: Available")
            print(f"    Resolution: {width}x{height}, FPS: {fps}")
            
            # Read a test frame
            ret, frame = cap.read()
            if ret:
                print(f"    Successfully read a frame of shape {frame.shape}")
            else:
                print(f"    Could not read a frame")
                
            available_cameras.append(i)
            cap.release()
        else:
            print(f"  Camera index {i}: Not available")
    
    return available_cameras

def check_v4l2_devices():
    """Check for V4L2 devices on Linux."""
    print("\nChecking V4L2 devices:")
    video_devices = [f for f in os.listdir('/dev') if f.startswith('video')]
    
    for device in sorted(video_devices):
        device_path = f"/dev/{device}"
        cap = cv2.VideoCapture(device_path)
        
        if cap.isOpened():
            # Get camera properties
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  {device_path}: Available")
            print(f"    Resolution: {width}x{height}, FPS: {fps}")
            
            # Try to read a frame
            ret, frame = cap.read()
            if ret:
                print(f"    Successfully read a frame of shape {frame.shape}")
            else:
                print(f"    Could not read a frame")
                
            cap.release()
        else:
            print(f"  {device_path}: Not available or no permissions")

def test_insta360_camera():
    """Test the Insta360 camera with different settings"""
    print("\nTesting Insta360 camera at /dev/video4 with special handling:")
    
    # Try opening with explicit V4L2 backend
    cap = cv2.VideoCapture("/dev/video4", cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("  Failed to open camera with V4L2 backend")
        return False
        
    print("  Camera opened successfully!")
    
    # Get initial properties
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"  Default resolution: {width}x{height}, FPS: {fps}")
    
    # Try different formats
    # Common Insta360 formats: MJPG, YUYV, H264
    print("  Trying different formats:")
    for format_code in [cv2.CAP_PROP_FOURCC]:
        fourcc = int(cap.get(format_code))
        fourcc_str = "".join([chr((fourcc >> 8*i) & 0xFF) for i in range(4)])
        print(f"  Current format: {fourcc_str}")
        
    # Try different resolutions
    resolutions = [
        (1920, 1080),  # Full HD
        (3840, 1920),  # 4K for 360 cameras
        (2880, 1440),  # 2.5K
        (1440, 720),   # HD
    ]
    
    for width, height in resolutions:
        print(f"\n  Testing resolution {width}x{height}:")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"  Set: {width}x{height}, Got: {actual_width}x{actual_height}")
        
        # Add a delay for camera to adjust
        time.sleep(0.5)
        
        # Try reading multiple frames (sometimes first frames fail)
        for i in range(5):
            print(f"  Attempt {i+1} to read frame...")
            try:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    print(f"  ✅ Success! Frame shape: {frame.shape}")
                    # Save the frame
                    filename = f"insta360_test_{width}x{height}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"  Frame saved as {filename}")
                    break
                else:
                    print("  ❌ Frame capture failed or empty frame")
                    time.sleep(0.5)  # Wait before retry
            except Exception as e:
                print(f"  ❌ Error reading frame: {e}")
                time.sleep(0.5)  # Wait before retry
    
    cap.release()
    print("Camera released")
    return True

if __name__ == "__main__":
    print("Checking for available cameras...")
    available_indices = list_available_cameras()
    
    if os.path.exists('/dev/video0'):  # Check if we're on a Linux system with V4L2
        check_v4l2_devices()
    
    test_insta360_camera()