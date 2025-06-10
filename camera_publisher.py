#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import sys
import os
import subprocess
import time
from threading import Thread, Lock

class Insta360Publisher(Node):
    def __init__(self):
        super().__init__("insta360_publisher")
        
        # Create publishers for camera views
        self.publisher = self.create_publisher(Image, "insta360/image_raw", 10)
        self.bridge = CvBridge()
        
        # Declare parameters
        self.declare_parameter('camera_device', '/dev/video4')  # Updated to your camera
        self.declare_parameter('width', 1920)  # Updated to 1920
        self.declare_parameter('height', 1080)  # Updated to 1080
        self.declare_parameter('fps', 30.0)
        self.declare_parameter('retry_count', 5)
        
        # Get parameter values
        self.device = self.get_parameter('camera_device').value
        self.width = self.get_parameter('width').value
        self.height = self.get_parameter('height').value
        self.fps = self.get_parameter('fps').value
        self.retry_count = self.get_parameter('retry_count').value
        
        # List available cameras
        self.list_available_cameras()
        
        # Connect to the camera
        self.connect_camera()
        
        # Set up timer for frame capture
        self.frame_lock = Lock()
        self.timer = self.create_timer(1.0/self.fps, self.capture_callback)
    
    def list_available_cameras(self):
        """List all available video devices"""
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                   capture_output=True, text=True)
            self.get_logger().info(f"Available cameras:\n{result.stdout}")
        except Exception as e:
            self.get_logger().warn(f"Could not list cameras: {e}")
    
    def connect_camera(self):
        """Connect to the Insta360 camera with robust error handling"""
        self.get_logger().info(f"Connecting to camera at {self.device}")
        
        # Open camera with V4L2 backend explicitly
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            self.get_logger().error(f"Failed to open camera at {self.device}")
            raise RuntimeError(f"Failed to open camera at {self.device}")
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        # Get actual camera properties
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        self.get_logger().info(f"Camera connected: {actual_width}x{actual_height} @ {actual_fps}fps")
        
        # Try to get a test frame
        for i in range(self.retry_count):
            self.get_logger().info(f"Attempting to read initial frame (attempt {i+1}/{self.retry_count})...")
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    self.get_logger().info(f"Successfully read initial frame: {frame.shape}")
                    break
                else:
                    self.get_logger().warn(f"Failed to read initial frame (attempt {i+1})")
                    time.sleep(0.5)
            except Exception as e:
                self.get_logger().error(f"Error reading initial frame: {e}")
                time.sleep(0.5)
        
    def capture_callback(self):
        """Capture and publish a frame"""
        with self.frame_lock:
            if not self.cap.isOpened():
                self.get_logger().error("Camera not open")
                return
                
            try:
                ret, frame = self.cap.read()
                
                if not ret or frame is None or frame.size == 0:
                    self.get_logger().warn("Failed to capture frame or empty frame received")
                    return
                    
                # Convert to ROS Image message
                msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = "insta360_camera"
                
                # Publish
                self.publisher.publish(msg)
                
            except cv2.error as e:
                self.get_logger().error(f"OpenCV error: {e}")
            except Exception as e:
                self.get_logger().error(f"Error capturing frame: {e}")
    
    def destroy_node(self):
        """Clean up when the node is shutting down"""
        self.get_logger().info("Shutting down camera...")
        with self.frame_lock:
            if hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = Insta360Publisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
