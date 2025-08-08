#!/usr/bin/env python3
"""
Obstacle Avoidance System using Camera and LIDAR
Author: AI Assistant
Description: Complete implementation for obstacle detection and avoidance
             using camera vision and LIDAR sensors
"""

import numpy as np
import cv2
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, List, Optional
import threading
import queue

# For LIDAR (assuming RPLidar or similar)
try:
    from rplidar import RPLidar
except ImportError:
    print("RPLidar library not installed. Install with: pip install rplidar")

# For robot control (example using GPIO for Raspberry Pi)
try:
    import RPi.GPIO as GPIO
except ImportError:
    print("RPi.GPIO not available. Using mock GPIO for testing")
    class GPIO:
        BCM = "BCM"
        OUT = "OUT"
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setup(pin, mode): pass
        @staticmethod
        def output(pin, value): pass
        @staticmethod
        def cleanup(): pass
        @staticmethod
        def PWM(pin, freq): 
            class MockPWM:
                def start(self, dc): pass
                def ChangeDutyCycle(self, dc): pass
                def stop(self): pass
            return MockPWM()

# ============================================================================
# Configuration and Data Structures
# ============================================================================

@dataclass
class ObstacleInfo:
    """Data structure for obstacle information"""
    distance: float  # Distance in meters
    angle: float     # Angle in degrees (-180 to 180)
    size: float      # Estimated size
    confidence: float # Detection confidence (0-1)
    source: str      # 'camera', 'lidar', or 'fusion'

@dataclass
class RobotConfig:
    """Robot configuration parameters"""
    # Safety thresholds
    min_safe_distance: float = 0.5  # meters
    emergency_stop_distance: float = 0.3  # meters
    
    # Movement parameters
    max_linear_speed: float = 1.0  # m/s
    max_angular_speed: float = 90.0  # degrees/s
    
    # Sensor parameters
    camera_fov: float = 60.0  # degrees
    lidar_max_range: float = 12.0  # meters
    
    # GPIO pins for motor control (example)
    motor_left_forward: int = 17
    motor_left_backward: int = 27
    motor_right_forward: int = 22
    motor_right_backward: int = 23
    motor_enable_left: int = 13
    motor_enable_right: int = 19

# ============================================================================
# Camera-based Obstacle Detection
# ============================================================================

class CameraObstacleDetector:
    """Handles camera-based obstacle detection using OpenCV"""
    
    def __init__(self, camera_index=0, resolution=(640, 480)):
        self.camera = cv2.VideoCapture(camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.resolution = resolution
        
        # Initialize background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True
        )
        
        # Parameters for depth estimation from monocular camera
        self.focal_length = 500  # Approximate focal length in pixels
        self.known_object_width = 0.5  # Assumed object width in meters
        
    def detect_obstacles(self, frame: np.ndarray) -> List[ObstacleInfo]:
        """Detect obstacles in camera frame"""
        obstacles = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        height, width = frame.shape[:2]
        center_x = width // 2
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter small contours
            if area < 500:
                continue
                
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Estimate distance using apparent size
            # (This is a simple approximation - use stereo vision for accuracy)
            if w > 0:
                distance = (self.known_object_width * self.focal_length) / w
                
                # Calculate angle from center
                object_center_x = x + w // 2
                angle = (object_center_x - center_x) * 60 / width  # Map to FOV
                
                # Calculate confidence based on contour properties
                solidity = area / (w * h)
                confidence = min(solidity * 1.5, 1.0)
                
                obstacles.append(ObstacleInfo(
                    distance=distance,
                    angle=angle,
                    size=w * h / (width * height),  # Normalized size
                    confidence=confidence,
                    source='camera'
                ))
                
        return obstacles
    
    def detect_moving_obstacles(self, frame: np.ndarray) -> List[ObstacleInfo]:
        """Detect moving obstacles using background subtraction"""
        obstacles = []
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours of moving objects
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        height, width = frame.shape[:2]
        center_x = width // 2
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1000:  # Filter small movements
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Estimate distance
            if w > 0:
                distance = (self.known_object_width * self.focal_length) / w
                object_center_x = x + w // 2
                angle = (object_center_x - center_x) * 60 / width
                
                obstacles.append(ObstacleInfo(
                    distance=distance,
                    angle=angle,
                    size=w * h / (width * height),
                    confidence=0.8,  # Moving objects have high priority
                    source='camera_motion'
                ))
                
        return obstacles
    
    def process_frame(self) -> Tuple[np.ndarray, List[ObstacleInfo]]:
        """Capture and process a single frame"""
        ret, frame = self.camera.read()
        if not ret:
            return None, []
            
        static_obstacles = self.detect_obstacles(frame)
        moving_obstacles = self.detect_moving_obstacles(frame)
        
        all_obstacles = static_obstacles + moving_obstacles
        
        # Draw obstacles on frame for visualization
        for obstacle in all_obstacles:
            # Convert back to pixel coordinates for drawing
            angle_normalized = obstacle.angle / 60  # Normalize to [-1, 1]
            x = int((angle_normalized + 1) * frame.shape[1] / 2)
            
            # Draw based on distance (closer = larger circle, redder color)
            radius = int(50 / max(obstacle.distance, 0.5))
            color_intensity = int(255 * (1 - min(obstacle.distance / 5, 1)))
            color = (color_intensity, 0, 255 - color_intensity)  # Blue to red
            
            cv2.circle(frame, (x, frame.shape[0] // 2), radius, color, -1)
            cv2.putText(frame, f"{obstacle.distance:.1f}m", 
                       (x - 20, frame.shape[0] // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        return frame, all_obstacles
    
    def release(self):
        """Release camera resources"""
        self.camera.release()

# ============================================================================
# LIDAR-based Obstacle Detection
# ============================================================================

class LidarObstacleDetector:
    """Handles LIDAR-based obstacle detection"""
    
    def __init__(self, port='/dev/ttyUSB0', max_range=12.0):
        self.port = port
        self.max_range = max_range
        self.lidar = None
        self.scan_data = []
        self.scanning = False
        self.scan_thread = None
        self.data_lock = threading.Lock()
        
        try:
            self.lidar = RPLidar(port)
            self.lidar.stop()
            time.sleep(0.1)
            self.lidar.start_motor()
        except:
            print(f"Could not connect to LIDAR on {port}")
            
    def start_scanning(self):
        """Start continuous LIDAR scanning in background thread"""
        if self.lidar and not self.scanning:
            self.scanning = True
            self.scan_thread = threading.Thread(target=self._scan_loop)
            self.scan_thread.daemon = True
            self.scan_thread.start()
            
    def _scan_loop(self):
        """Background scanning loop"""
        try:
            for scan in self.lidar.iter_scans():
                if not self.scanning:
                    break
                    
                with self.data_lock:
                    self.scan_data = scan
        except Exception as e:
            print(f"LIDAR scanning error: {e}")
            
    def stop_scanning(self):
        """Stop LIDAR scanning"""
        self.scanning = False
        if self.scan_thread:
            self.scan_thread.join(timeout=1)
        if self.lidar:
            self.lidar.stop()
            self.lidar.stop_motor()
            
    def process_scan(self) -> List[ObstacleInfo]:
        """Process LIDAR scan data to detect obstacles"""
        obstacles = []
        
        with self.data_lock:
            scan_copy = list(self.scan_data)
            
        if not scan_copy:
            return obstacles
            
        # Group nearby points into obstacles
        sectors = {}  # Group by 10-degree sectors
        
        for quality, angle, distance in scan_copy:
            if quality < 10:  # Filter low quality readings
                continue
                
            distance_m = distance / 1000.0  # Convert mm to meters
            
            if distance_m > self.max_range or distance_m < 0.1:
                continue
                
            # Group into 10-degree sectors
            sector = int(angle // 10) * 10
            
            if sector not in sectors:
                sectors[sector] = []
            sectors[sector].append(distance_m)
            
        # Create obstacle info for each sector with detected objects
        for sector, distances in sectors.items():
            if distances:
                min_distance = min(distances)
                avg_distance = np.mean(distances)
                
                # Convert sector to angle relative to front (-180 to 180)
                angle = sector - 180 if sector > 180 else sector
                
                # Confidence based on number of points
                confidence = min(len(distances) / 10.0, 1.0)
                
                obstacles.append(ObstacleInfo(
                    distance=min_distance,
                    angle=angle,
                    size=len(distances) / 360.0,  # Normalized by full scan
                    confidence=confidence,
                    source='lidar'
                ))
                
        return obstacles
    
    def get_obstacle_map(self) -> np.ndarray:
        """Generate 2D obstacle map from LIDAR data"""
        # Create a 2D grid (top-down view)
        grid_size = 100  # 100x100 grid
        grid = np.zeros((grid_size, grid_size))
        
        with self.data_lock:
            scan_copy = list(self.scan_data)
            
        for quality, angle, distance in scan_copy:
            if quality < 10:
                continue
                
            distance_m = distance / 1000.0
            if distance_m > self.max_range:
                continue
                
            # Convert polar to cartesian
            x = distance_m * np.cos(np.radians(angle))
            y = distance_m * np.sin(np.radians(angle))
            
            # Map to grid coordinates
            grid_x = int((x + self.max_range) * grid_size / (2 * self.max_range))
            grid_y = int((y + self.max_range) * grid_size / (2 * self.max_range))
            
            if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
                grid[grid_y, grid_x] = 1
                
        return grid

# ============================================================================
# Sensor Fusion
# ============================================================================

class SensorFusion:
    """Fuses camera and LIDAR data for robust obstacle detection"""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.obstacle_history = deque(maxlen=10)
        
    def fuse_obstacles(self, 
                       camera_obstacles: List[ObstacleInfo],
                       lidar_obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
        """Fuse obstacles from multiple sensors"""
        fused_obstacles = []
        
        # Match obstacles from different sensors
        matched_lidar = set()
        
        for cam_obs in camera_obstacles:
            best_match = None
            best_score = float('inf')
            
            for i, lidar_obs in enumerate(lidar_obstacles):
                if i in matched_lidar:
                    continue
                    
                # Calculate matching score based on angle and distance
                angle_diff = abs(cam_obs.angle - lidar_obs.angle)
                distance_diff = abs(cam_obs.distance - lidar_obs.distance)
                
                # Weighted score
                score = angle_diff / 30.0 + distance_diff / 2.0
                
                if score < best_score and score < 1.0:  # Threshold for matching
                    best_score = score
                    best_match = i
                    
            if best_match is not None:
                # Fuse the matched obstacles
                matched_lidar.add(best_match)
                lidar_obs = lidar_obstacles[best_match]
                
                # LIDAR is more accurate for distance, camera for lateral position
                fused_obs = ObstacleInfo(
                    distance=lidar_obs.distance,  # Trust LIDAR for distance
                    angle=(cam_obs.angle + lidar_obs.angle) / 2,  # Average angle
                    size=max(cam_obs.size, lidar_obs.size),
                    confidence=min(cam_obs.confidence + lidar_obs.confidence, 1.0),
                    source='fusion'
                )
                fused_obstacles.append(fused_obs)
            else:
                # No match found, keep camera obstacle
                fused_obstacles.append(cam_obs)
                
        # Add unmatched LIDAR obstacles
        for i, lidar_obs in enumerate(lidar_obstacles):
            if i not in matched_lidar:
                fused_obstacles.append(lidar_obs)
                
        # Apply temporal filtering
        self.obstacle_history.append(fused_obstacles)
        
        return self._temporal_filter(fused_obstacles)
    
    def _temporal_filter(self, current_obstacles: List[ObstacleInfo]) -> List[ObstacleInfo]:
        """Apply temporal filtering to reduce noise"""
        if len(self.obstacle_history) < 3:
            return current_obstacles
            
        filtered_obstacles = []
        
        for obs in current_obstacles:
            # Count similar obstacles in history
            persistence_count = 0
            
            for historical_frame in self.obstacle_history:
                for hist_obs in historical_frame:
                    angle_similar = abs(hist_obs.angle - obs.angle) < 15
                    distance_similar = abs(hist_obs.distance - obs.distance) < 1.0
                    
                    if angle_similar and distance_similar:
                        persistence_count += 1
                        break
                        
            # Keep obstacle if it persists across frames
            if persistence_count >= len(self.obstacle_history) // 2:
                obs.confidence = min(obs.confidence * 1.2, 1.0)
                filtered_obstacles.append(obs)
                
        return filtered_obstacles

# ============================================================================
# Path Planning and Avoidance
# ============================================================================

class ObstacleAvoidanceController:
    """Main controller for obstacle avoidance"""
    
    def __init__(self, config: RobotConfig):
        self.config = config
        self.current_speed = 0.0
        self.current_turn_rate = 0.0
        
        # Initialize motor control
        self._init_motors()
        
    def _init_motors(self):
        """Initialize GPIO for motor control"""
        GPIO.setmode(GPIO.BCM)
        
        # Setup motor pins
        GPIO.setup(self.config.motor_left_forward, GPIO.OUT)
        GPIO.setup(self.config.motor_left_backward, GPIO.OUT)
        GPIO.setup(self.config.motor_right_forward, GPIO.OUT)
        GPIO.setup(self.config.motor_right_backward, GPIO.OUT)
        GPIO.setup(self.config.motor_enable_left, GPIO.OUT)
        GPIO.setup(self.config.motor_enable_right, GPIO.OUT)
        
        # Setup PWM for speed control
        self.pwm_left = GPIO.PWM(self.config.motor_enable_left, 100)
        self.pwm_right = GPIO.PWM(self.config.motor_enable_right, 100)
        self.pwm_left.start(0)
        self.pwm_right.start(0)
        
    def compute_avoidance_command(self, 
                                  obstacles: List[ObstacleInfo]) -> Tuple[float, float]:
        """Compute speed and steering commands to avoid obstacles"""
        
        if not obstacles:
            # No obstacles, move forward
            return self.config.max_linear_speed, 0.0
            
        # Find closest obstacle
        closest = min(obstacles, key=lambda o: o.distance)
        
        # Emergency stop if too close
        if closest.distance < self.config.emergency_stop_distance:
            return 0.0, 0.0
            
        # Calculate repulsive forces from all obstacles
        total_force_x = 0.0
        total_force_y = 0.0
        
        for obstacle in obstacles:
            if obstacle.distance > self.config.min_safe_distance * 3:
                continue  # Ignore distant obstacles
                
            # Repulsive force inversely proportional to distance
            force_magnitude = 1.0 / max(obstacle.distance, 0.1)
            force_magnitude *= obstacle.confidence
            
            # Convert to cartesian forces
            angle_rad = np.radians(obstacle.angle)
            force_x = -force_magnitude * np.sin(angle_rad)
            force_y = -force_magnitude * np.cos(angle_rad)
            
            total_force_x += force_x
            total_force_y += force_y
            
        # Add attractive force towards goal (straight ahead)
        goal_force = 1.0
        total_force_y += goal_force
        
        # Convert forces to speed and steering
        desired_speed = np.clip(total_force_y, 0, 1) * self.config.max_linear_speed
        
        # Reduce speed based on closest obstacle
        speed_reduction = np.exp(-2 * (closest.distance - self.config.min_safe_distance))
        speed_reduction = np.clip(speed_reduction, 0, 1)
        desired_speed *= (1 - speed_reduction)
        
        # Calculate steering angle
        desired_turn = np.arctan2(total_force_x, total_force_y)
        desired_turn_rate = np.degrees(desired_turn)
        desired_turn_rate = np.clip(desired_turn_rate, 
                                    -self.config.max_angular_speed,
                                    self.config.max_angular_speed)
        
        return desired_speed, desired_turn_rate
    
    def apply_dynamic_window(self, 
                            desired_speed: float,
                            desired_turn: float,
                            obstacles: List[ObstacleInfo]) -> Tuple[float, float]:
        """Apply Dynamic Window Approach for smooth control"""
        
        # Define acceleration limits
        max_accel = 0.5  # m/s^2
        max_turn_accel = 45  # degrees/s^2
        dt = 0.1  # Time step
        
        # Calculate reachable velocities
        min_speed = max(0, self.current_speed - max_accel * dt)
        max_speed = min(self.config.max_linear_speed, 
                       self.current_speed + max_accel * dt)
        
        min_turn = max(-self.config.max_angular_speed,
                      self.current_turn_rate - max_turn_accel * dt)
        max_turn = min(self.config.max_angular_speed,
                      self.current_turn_rate + max_turn_accel * dt)
        
        # Evaluate different velocity combinations
        best_speed = self.current_speed
        best_turn = self.current_turn_rate
        best_score = -float('inf')
        
        for speed in np.linspace(min_speed, max_speed, 5):
            for turn in np.linspace(min_turn, max_turn, 5):
                # Simulate trajectory
                score = self._evaluate_trajectory(speed, turn, obstacles)
                
                # Add preference for desired velocity
                speed_diff = abs(speed - desired_speed)
                turn_diff = abs(turn - desired_turn)
                score -= 0.1 * speed_diff + 0.05 * turn_diff
                
                if score > best_score:
                    best_score = score
                    best_speed = speed
                    best_turn = turn
                    
        return best_speed, best_turn
    
    def _evaluate_trajectory(self, 
                           speed: float, 
                           turn_rate: float,
                           obstacles: List[ObstacleInfo]) -> float:
        """Evaluate a trajectory for collision risk"""
        
        # Simulate trajectory for next 2 seconds
        dt = 0.1
        steps = 20
        
        x, y, theta = 0, 0, 0
        min_clearance = float('inf')
        
        for _ in range(steps):
            # Update position
            theta += np.radians(turn_rate) * dt
            x += speed * np.sin(theta) * dt
            y += speed * np.cos(theta) * dt
            
            # Check clearance to obstacles
            for obstacle in obstacles:
                # Convert obstacle to cartesian
                obs_angle = np.radians(obstacle.angle)
                obs_x = obstacle.distance * np.sin(obs_angle)
                obs_y = obstacle.distance * np.cos(obs_angle)
                
                # Calculate distance
                distance = np.sqrt((x - obs_x)**2 + (y - obs_y)**2)
                min_clearance = min(min_clearance, distance)
                
        # Score based on clearance and forward progress
        clearance_score = min_clearance
        progress_score = y  # Forward progress
        
        return clearance_score + 0.5 * progress_score
    
    def execute_control(self, speed: float, turn_rate: float):
        """Execute motor control commands"""
        
        # Update current state
        self.current_speed = speed
        self.current_turn_rate = turn_rate
        
        # Convert to differential drive commands
        # (assuming two-wheel differential drive robot)
        wheel_base = 0.3  # meters between wheels
        
        # Calculate wheel speeds
        left_speed = speed - (turn_rate * wheel_base / 2)
        right_speed = speed + (turn_rate * wheel_base / 2)
        
        # Normalize to PWM duty cycle (0-100)
        max_wheel_speed = self.config.max_linear_speed + \
                         (self.config.max_angular_speed * wheel_base / 2)
        
        left_duty = abs(left_speed / max_wheel_speed * 100)
        right_duty = abs(right_speed / max_wheel_speed * 100)
        
        # Set motor directions
        if left_speed >= 0:
            GPIO.output(self.config.motor_left_forward, GPIO.HIGH)
            GPIO.output(self.config.motor_left_backward, GPIO.LOW)
        else:
            GPIO.output(self.config.motor_left_forward, GPIO.LOW)
            GPIO.output(self.config.motor_left_backward, GPIO.HIGH)
            
        if right_speed >= 0:
            GPIO.output(self.config.motor_right_forward, GPIO.HIGH)
            GPIO.output(self.config.motor_right_backward, GPIO.LOW)
        else:
            GPIO.output(self.config.motor_right_forward, GPIO.LOW)
            GPIO.output(self.config.motor_right_backward, GPIO.HIGH)
            
        # Set speeds
        self.pwm_left.ChangeDutyCycle(min(left_duty, 100))
        self.pwm_right.ChangeDutyCycle(min(right_duty, 100))
        
    def stop(self):
        """Stop all motors"""
        self.pwm_left.ChangeDutyCycle(0)
        self.pwm_right.ChangeDutyCycle(0)
        GPIO.output(self.config.motor_left_forward, GPIO.LOW)
        GPIO.output(self.config.motor_left_backward, GPIO.LOW)
        GPIO.output(self.config.motor_right_forward, GPIO.LOW)
        GPIO.output(self.config.motor_right_backward, GPIO.LOW)
        
    def cleanup(self):
        """Clean up GPIO resources"""
        self.stop()
        self.pwm_left.stop()
        self.pwm_right.stop()
        GPIO.cleanup()

# ============================================================================
# Main Application
# ============================================================================

class ObstacleAvoidanceSystem:
    """Main system orchestrating all components"""
    
    def __init__(self, config: RobotConfig = None):
        self.config = config or RobotConfig()
        
        # Initialize components
        print("Initializing Obstacle Avoidance System...")
        
        self.camera_detector = CameraObstacleDetector()
        self.lidar_detector = LidarObstacleDetector()
        self.sensor_fusion = SensorFusion(self.config)
        self.controller = ObstacleAvoidanceController(self.config)
        
        self.running = False
        self.visualization_enabled = True
        
    def run(self):
        """Main control loop"""
        print("Starting obstacle avoidance system...")
        
        # Start LIDAR scanning
        self.lidar_detector.start_scanning()
        
        self.running = True
        
        try:
            while self.running:
                # Get sensor data
                frame, camera_obstacles = self.camera_detector.process_frame()
                lidar_obstacles = self.lidar_detector.process_scan()
                
                # Fuse sensor data
                fused_obstacles = self.sensor_fusion.fuse_obstacles(
                    camera_obstacles, lidar_obstacles
                )
                
                # Compute avoidance commands
                desired_speed, desired_turn = self.controller.compute_avoidance_command(
                    fused_obstacles
                )
                
                # Apply dynamic window for smooth control
                speed, turn = self.controller.apply_dynamic_window(
                    desired_speed, desired_turn, fused_obstacles
                )
                
                # Execute control
                self.controller.execute_control(speed, turn)
                
                # Visualization
                if self.visualization_enabled and frame is not None:
                    self._visualize(frame, fused_obstacles, speed, turn)
                    
                # Check for exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nStopping system...")
        finally:
            self.cleanup()
            
    def _visualize(self, frame: np.ndarray, 
                   obstacles: List[ObstacleInfo],
                   speed: float, turn: float):
        """Visualize system state"""
        
        # Add status text
        cv2.putText(frame, f"Speed: {speed:.2f} m/s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Turn: {turn:.1f} deg/s", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Obstacles: {len(obstacles)}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw obstacle indicators
        height, width = frame.shape[:2]
        
        # Draw safety zones
        cv2.line(frame, (0, height - 50), (width, height - 50),
                (255, 255, 0), 1)  # Warning line
        cv2.line(frame, (0, height - 100), (width, height - 100),
                (0, 255, 255), 1)  # Safe zone line
        
        # Show frame
        cv2.imshow('Obstacle Avoidance System', frame)
        
        # Create top-down LIDAR view
        lidar_map = self.lidar_detector.get_obstacle_map()
        if lidar_map is not None:
            lidar_display = cv2.resize(lidar_map * 255, (300, 300))
            lidar_display = cv2.applyColorMap(
                lidar_display.astype(np.uint8), cv2.COLORMAP_JET
            )
            cv2.imshow('LIDAR Map', lidar_display)
            
    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up...")
        self.running = False
        self.controller.cleanup()
        self.camera_detector.release()
        self.lidar_detector.stop_scanning()
        cv2.destroyAllWindows()
        
    def emergency_stop(self):
        """Emergency stop function"""
        print("EMERGENCY STOP!")
        self.running = False
        self.controller.stop()

# ============================================================================
# Utility Functions
# ============================================================================

def test_sensors():
    """Test sensor functionality"""
    print("Testing sensors...")
    
    # Test camera
    cam = CameraObstacleDetector()
    ret, frame = cam.camera.read()
    if ret:
        print("✓ Camera working")
    else:
        print("✗ Camera not working")
    cam.release()
    
    # Test LIDAR
    try:
        lidar = LidarObstacleDetector()
        lidar.start_scanning()
        time.sleep(2)
        obstacles = lidar.process_scan()
        if obstacles:
            print(f"✓ LIDAR working - detected {len(obstacles)} obstacles")
        else:
            print("✓ LIDAR working - no obstacles detected")
        lidar.stop_scanning()
    except:
        print("✗ LIDAR not connected")
        
def calibrate_camera():
    """Interactive camera calibration"""
    print("Camera Calibration Mode")
    print("Place an object of known size at known distance")
    print("Press 'c' to capture, 'q' to quit")
    
    cam = CameraObstacleDetector()
    
    while True:
        ret, frame = cam.camera.read()
        if not ret:
            break
            
        cv2.imshow('Calibration', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            # Capture calibration frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(
                edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            if contours:
                largest = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest)
                
                print(f"Object width in pixels: {w}")
                actual_width = float(input("Enter actual width in meters: "))
                distance = float(input("Enter distance in meters: "))
                
                focal_length = (w * distance) / actual_width
                print(f"Calculated focal length: {focal_length}")
                
        elif key == ord('q'):
            break
            
    cam.release()
    cv2.destroyAllWindows()

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Obstacle Avoidance System using Camera and LIDAR"
    )
    parser.add_argument('--test', action='store_true',
                       help='Test sensor functionality')
    parser.add_argument('--calibrate', action='store_true',
                       help='Calibrate camera')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    if args.test:
        test_sensors()
    elif args.calibrate:
        calibrate_camera()
    else:
        # Run main system
        config = RobotConfig()
        system = ObstacleAvoidanceSystem(config)
        
        if args.no_viz:
            system.visualization_enabled = False
            
        try:
            system.run()
        except Exception as e:
            print(f"Error: {e}")
            system.cleanup()