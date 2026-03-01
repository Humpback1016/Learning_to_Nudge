#!/usr/bin/env python3

import argparse
import numpy as np
import torch
from collections import deque
import time
import signal
import sys
import threading
import json
from datetime import datetime
import os

# ROS2 imports
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

from frankx import Robot, Affine, PathMotion, JointMotion

from dcbf_real_exp.models import NCBF, Normalizer

frankx_robot = None


class DataRecorder:
    """Data recorder for trajectory collection"""
    
    def __init__(self, output_dir="real_robot_trajectories"):
        self.output_dir = output_dir
        self.current_trajectory = []
        self.step_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def start_trajectory(self):
        """Start recording a new trajectory"""
        self.current_trajectory = []
        self.step_count = 0
        print("Started recording new trajectory")
    
    def record_timestep(self, ncbf_system):
        """Record current timestep data"""

        # Get EE position
        ee_x, ee_y, ee_z = ncbf_system.get_ee_position()
        if ee_x is None:
            print("Warning: Cannot get EE position, skipping this timestep")
            return
        
        # Get robot joint positions
        robot_joints = self.get_joint_positions()
        
        # Get all objects state
        objects_state = []
        for tag_id in ncbf_system.tag_ids:
            if ncbf_system.is_tag_data_valid(tag_id):
                history = ncbf_system.objects[tag_id]['history']
                if len(history) > 0:
                    latest = history[-1]  # [tilt, x, y, z]
                    objects_state.append({
                        "tag_id": int(tag_id),
                        "name": f"object_tag_{tag_id}",
                        "position": {
                            "x": float(latest[1]),
                            "y": float(latest[2]),
                            "z": float(latest[3])
                        },
                        "tilt_cost": float(latest[0])
                    })
        
        # Compute safety status
        is_safe = self.compute_is_safe(ncbf_system)
        
        # Create timestep data
        timestep_data = {
            "timestamp": time.time(),
            "step_index": self.step_count,
            "robot_joints": robot_joints,
            "ee_state": {
                "x": float(ee_x),
                "y": float(ee_y),
                "z": float(ee_z)
            },
            "objects_state": objects_state,
            "is_safe": is_safe
        }
        
        self.current_trajectory.append(timestep_data)
        self.step_count += 1
            
    def get_joint_positions(self):
        """Get robot joint positions from frankx"""
        global frankx_robot

        robot_state = frankx_robot.robot.readOnce()
        return [float(q) for q in robot_state.q]

    def compute_is_safe(self, ncbf_system):
        """Compute overall safety status"""
        for tag_id in ncbf_system.tag_ids:
            if ncbf_system.is_tag_data_valid(tag_id):
                history = ncbf_system.objects[tag_id]['history']
                if len(history) > 0:
                    tilt_angle = history[-1][0]
                    tilt_degrees = np.degrees(tilt_angle)
                    if tilt_degrees > ncbf_system.max_tipping_angle:
                        return False
        return True
    
    def save_trajectory(self, filename=None):
        """Save trajectory to JSON file"""

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"trajectory_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.current_trajectory, f, indent=2)
        
        print(f"Trajectory saved to {filepath}")
        print(f"Total timesteps recorded: {len(self.current_trajectory)}")
        return filepath

    
    def get_trajectory_stats(self):
        """Get statistics about the recorded trajectory"""
        if len(self.current_trajectory) == 0:
            return None
        
        num_objects = len(self.current_trajectory[0]['objects_state']) if self.current_trajectory else 0
        safe_count = sum(1 for step in self.current_trajectory if step['is_safe'])
        
        return {
            'total_steps': len(self.current_trajectory),
            'num_objects': num_objects,
            'safe_steps': safe_count,
            'unsafe_steps': len(self.current_trajectory) - safe_count
        }


class SimplifiedNCBFSystem(Node):
    def __init__(self, robot_ip='192.168.1.116', tag_ids=None):
        super().__init__('ncbf_controller')
        
        self.history_horizon = 6  
        self.cbf_threshold = 0.0  
        self.max_tipping_angle = 15.0  
        self.marker_length = 0.03  
        self.current_velocity = 0.01  
        
        self.push_height = 0.255
        self.goal_threshold = 0.01
        
        # Load models
        self.ncbf_model, self.h_obj_normalizer, self.ee_normalizer = self.load_models()
        
        # Multiple objects tracking
        self.tag_ids = tag_ids if tag_ids else [2]
        self.objects = {}
        for tag_id in self.tag_ids:
            self.objects[tag_id] = {
                'pose': None,
                'history': deque(maxlen=self.history_horizon),
                'last_update': None,
                'received': False
            }
        
        # Primary object for control (default to first tag)
        self.primary_tag_id = self.tag_ids[0]
        
        self.ee_position = None
        
        # Subscribe to unified AprilTag poses (already in base frame!)
        self.pose_subscribers = {}
        for tag_id in self.tag_ids:
            topic_name = f'/apriltag/tag_{tag_id}/pose'
            self.pose_subscribers[tag_id] = self.create_subscription(
                PoseStamped,
                topic_name,
                lambda msg, tid=tag_id: self.apriltag_pose_callback(msg, tid),
                10
            )
            self.get_logger().info(f'Subscribed to topic: {topic_name}')
        
        # Data validity
        self.apriltag_timeout = 2.0
        
        # Initialize Frankx robot
        self.robot_ip = robot_ip
        self.init_robot()
        
        # Target point
        self.target_point = self.generate_target_point()
        self.get_logger().info(f'Target point set to: [{self.target_point[0]:.3f}, {self.target_point[1]:.3f}, {self.target_point[2]:.3f}]')
        
        # Statistics
        self.total_steps = 0
        self.safety_violations = 0
        self.ncbf_preventions = 0
        
        # Data recorder
        self.data_recorder = DataRecorder()
        
        self.get_logger().info(f'Simplified NCBF system with ROS2 integration started')
        self.get_logger().info(f'Tracking tags: {self.tag_ids}')
        self.get_logger().info(f'Primary tag for control: {self.primary_tag_id}')
    
    def load_models(self):
        """Load NCBF model and normalizers"""
        self.get_logger().info('Loading NCBF model and normalizers')
        
        model_path = "dcbf_real_exp/results_sigma_0.02_his6/model_batch_47.pt"
        h_obj_normalizer_path = "dcbf_real_exp/results_sigma_0.02_his6/local_normalizer"
        ee_normalizer_path = "dcbf_real_exp/results_sigma_0.02_his6/arm_normalizer"

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        ncbf_model = NCBF.load(
            path=model_path,
            hiddens=[64, 64], 
            seq_hiddens=[64],
            map_location=device
        )
        
        h_obj_normalizer = Normalizer(input_size=4)
        h_obj_normalizer.load_model(h_obj_normalizer_path)
        
        ee_normalizer = Normalizer(input_size=2)
        ee_normalizer.load_model(ee_normalizer_path)
        
        ncbf_model = ncbf_model.to(device)
        ncbf_model.eval()
        
        self.get_logger().info(f'Model loaded successfully, using device: {device}')
        return ncbf_model, h_obj_normalizer, ee_normalizer
    
    def init_robot(self):
        """Initialize Frankx robot"""
        global frankx_robot

        self.get_logger().info(f'Connecting to Franka Emika robot IP: {self.robot_ip}')
        frankx_robot = Robot(self.robot_ip)
        
        frankx_robot.set_default_behavior()
        frankx_robot.recover_from_errors()
        frankx_robot.set_dynamic_rel(0.15)
        frankx_robot.move(JointMotion([-1.811944, 1.179108, 1.757100, -2.14162, -1.143369, 1.633046, -0.432171]))
        # frankx_robot.move(JointMotion([-1.811944, 1.179108, 1.757100, -2.14162, -1.143369, 1.633046, 2.709421]))
        # frankx_robot.move(JointMotion([0.0,-0.3,0.0,-2.0,0.0,2.5,0.7]))
        self.get_logger().info('Robot connected successfully!')
        
        ee_x, ee_y, ee_z = self.get_ee_position()
        if ee_x is not None:
            self.ee_position = np.array([ee_x, ee_y, ee_z])
            self.get_logger().info(f'Initial EE position: X={ee_x:.4f}, Y={ee_y:.4f}, Z={ee_z:.4f}')

    def get_ee_position(self):
        """Get robot end-effector position"""
        global frankx_robot

        current_pose = frankx_robot.current_pose()
        position = current_pose.translation()
        return position[0], position[1], position[2]


    def marker_to_center_position(self, marker_position, quaternion, can_height=0.15):
        """
        Transform ArUco marker position (top of can) to can center position
        
        Args:
            marker_position: [x, y, z] position of marker in world frame
            quaternion: [x, y, z, w] orientation of marker
            can_height: height of the cylindrical can (default 0.15m for standard soda can)
        
        Returns:
            center_position: [x, y, z] position of can center in world frame
        """
        x, y, z, w = quaternion
        
        # Extract local Z-axis from quaternion (3rd column of rotation matrix)
        local_z_x = 2.0 * (x*z + w*y)
        local_z_y = 2.0 * (y*z - w*x)
        local_z_z = 1.0 - 2.0 * (x*x + y*y)
        
        local_z_axis = np.array([local_z_x, local_z_y, local_z_z])
        
        # Offset from marker to center = half can height along local Z-axis
        offset = (can_height / 2.0) * local_z_axis
        
        # Center is marker position minus the offset
        center_position = marker_position - offset
        
        return center_position
    
    def apriltag_pose_callback(self, msg, tag_id):
        """Callback function - pose is already in panda_link0 frame from fusion node"""

        # Directly extract position (already in base frame!)
        position = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        
        # Extract quaternion
        orientation = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w
        ])
        
        # Update object pose for this specific tag
        self.update_tag_pose(tag_id, position, orientation)
        
        # Update flags for this tag
        self.objects[tag_id]['received'] = True
        self.objects[tag_id]['last_update'] = self.get_clock().now()
        
        # Log first reception for each tag
        if not hasattr(self, f'_first_pose_logged_{tag_id}'):
            self.get_logger().info(f'✓ First pose received for tag {tag_id} (in base frame)')
            self.get_logger().info(f'  Position: [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]')
            setattr(self, f'_first_pose_logged_{tag_id}', True)

    
    def is_tag_data_valid(self, tag_id):
        """Check if specific tag data is valid and not too old"""
        if tag_id not in self.objects:
            return False
        
        tag_data = self.objects[tag_id]
        
        if not tag_data['received'] or tag_data['last_update'] is None:
            return False
        
        current_time = self.get_clock().now()
        time_diff = (current_time - tag_data['last_update']).nanoseconds / 1e9
        
        if time_diff > self.apriltag_timeout:
            self.get_logger().warn(f'Tag {tag_id} data is stale ({time_diff:.2f}s old)')
            return False
        
        return True
    
    def are_all_tags_valid(self):
        """Check if all tracked tags have valid data"""
        return all(self.is_tag_data_valid(tag_id) for tag_id in self.tag_ids)
    
    def get_valid_tags(self):
        """Get list of tag IDs with valid data"""
        return [tag_id for tag_id in self.tag_ids if self.is_tag_data_valid(tag_id)]
    
    #this z position for object is not fixed

    # def update_tag_pose(self, tag_id, position, orientation):
    #     """Update received AprilTag pose information for specific tag"""
    #     if tag_id not in self.objects:
    #         self.get_logger().warn(f'Received pose for untracked tag {tag_id}')
    #         return
        
    #     self.objects[tag_id]['pose'] = np.concatenate([position, orientation])
        
    #     tilt_angle = self.calculate_tilt_angle(orientation)
        
    #     history = self.objects[tag_id]['history']
        
    #     if len(history) < self.history_horizon:
    #         if len(history) == 0:
    #             initial_state = [tilt_angle, position[0], position[1], position[2]]
    #             for _ in range(self.history_horizon):
    #                 history.append(initial_state)
    #         else:
    #             current_state = [tilt_angle, position[0], position[1], position[2]]
    #             history.append(current_state)
    #     else:
    #         current_state = [tilt_angle, position[0], position[1], position[2]]
    #         history.append(current_state)

    #this z position for object is fixed

    def update_tag_pose(self, tag_id, position, orientation):
        """Update received AprilTag pose information for specific tag"""
        if tag_id not in self.objects:
            self.get_logger().warn(f'Received pose for untracked tag {tag_id}')
            return
        
        self.objects[tag_id]['pose'] = np.concatenate([position, orientation])
        
        tilt_angle = self.calculate_tilt_angle(orientation)
        
        history = self.objects[tag_id]['history']
        
        # fixed Z coordinate to 0.22 meters (for NCBF model input)
        fixed_z = 0.22
        
        if len(history) < self.history_horizon:
            if len(history) == 0:
                initial_state = [tilt_angle, position[0], position[1], fixed_z] 
                for _ in range(self.history_horizon):
                    history.append(initial_state)
            else:
                # current_state = [tilt_angle, position[0], position[1], fixed_z]
                center_pos = self.marker_to_center_position(position, orientation, can_height=0.22)
                current_state = [tilt_angle, center_pos[0], center_pos[1], fixed_z]
                history.append(current_state)
        else:
            # current_state = [tilt_angle, position[0], position[1], fixed_z]
            center_pos = self.marker_to_center_position(position, orientation, can_height=0.22)
            current_state = [tilt_angle, center_pos[0], center_pos[1], fixed_z]
            history.append(current_state)
                
    # def calculate_tilt_angle(self, quaternion):
    #     """Calculate tilt angle from quaternion"""
    #     x, y, z, w = quaternion
        
    #     sinr_cosp = 2.0 * (w * x + y * z)
    #     cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    #     roll = np.arctan2(sinr_cosp, cosr_cosp)
        
    #     sinp = 2.0 * (w * y - z * x)
    #     if abs(sinp) >= 1:
    #         pitch = np.copysign(np.pi / 2, sinp)
    #     else:
    #         pitch = np.arcsin(sinp)
        
    #     tilt = np.sqrt(roll*roll + pitch*pitch)
    #     return tilt

    def calculate_tilt_angle(self, quaternion):
        """Calculate tilt angle as the angle between object Z-axis and world Z-axis"""
        x, y, z, w = quaternion

        object_z_x = 2.0 * (x*z + w*y)
        object_z_y = 2.0 * (y*z - w*x)
        object_z_z = 1.0 - 2.0 * (x*x + y*y)
        
        dot_product = np.clip(object_z_z, -1.0, 1.0)
        tilt = np.arccos(np.abs(dot_product))
        
        return tilt
   
    def generate_target_point(self):
        """Generate target point"""
        random_x = 0.50 #0.65 0.45
        random_y = 0.14 #0.05 0.20

        target_world_pos = np.array([
            random_x,
            random_y,
            self.push_height  
        ])
        
        return target_world_pos
    
    def evaluate_ncbf_safety(self, next_position, tag_id=None, return_value=False):
        """Evaluate NCBF safety of next position for specific tag
        
        Args:
            next_position: Next position to evaluate
            tag_id: Tag ID to check
            return_value: If True, return (is_safe, cbf_value) tuple  
        
        Returns:
            bool or tuple: Safety status, or (is_safe, cbf_value) if return_value=True
        """
        if tag_id is None:
            tag_id = self.primary_tag_id
        
        if tag_id not in self.objects:
            return True if not return_value else (True, 0.0)  # ← 修改
        
        history = self.objects[tag_id]['history']
        
        if len(history) < self.history_horizon:
            return True if not return_value else (True, 0.0)  # ← 修改
        
        history_array = np.array(list(history))
        base_pos = history_array[0][1:4]
        
        rel_history = []
        for point in history_array:
            obj_tilt = point[0]
            obj_pos = point[1:4]
            rel_pos = obj_pos - base_pos
            rel_history.append([obj_tilt, rel_pos[0], rel_pos[1], rel_pos[2]])
        
        rel_history = np.array(rel_history)
        rel_robot_pos = np.array([next_position[0], next_position[1]]) - base_pos[:2]
        
        normalized_rel_history = self.h_obj_normalizer(rel_history, update=False)
        normalized_rel_robot_pos = self.ee_normalizer(np.array([rel_robot_pos]), update=False)[0]
        
        device = next(self.ncbf_model.parameters()).device
        history_tensor = torch.tensor(normalized_rel_history, dtype=torch.float32, device=device).unsqueeze(0)
        position_tensor = torch.tensor(normalized_rel_robot_pos, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            cbf_value = self.ncbf_model([history_tensor, position_tensor]).cpu().detach().numpy()[0, 0]  # ← 改为 [0, 0]
        
        is_safe = cbf_value < self.cbf_threshold
        
        if return_value:
            return bool(is_safe), float(cbf_value)
        else:
            return bool(is_safe)
        
    def evaluate_ncbf_safety_all_objects(self, next_position):
        """Evaluate NCBF safety considering all tracked objects"""
        valid_tags = self.get_valid_tags()
        
        if not valid_tags:
            return True
        
        for tag_id in valid_tags:
            if not self.evaluate_ncbf_safety(next_position, tag_id):
                return False
        
        return True
    
    def select_safe_action(self, current_pos, target_pos, num_samples=32):
        """Select safe action considering all tracked objects"""
        nominal_direction = target_pos - current_pos
        direction_norm = np.linalg.norm(nominal_direction[:2])
        
        if direction_norm < 0.001:
            return np.zeros(3), True
        
        normalized_direction = nominal_direction.copy()
        normalized_direction[:2] = nominal_direction[:2] / direction_norm * self.current_velocity
        normalized_direction[2] = 0
        
        next_pos = current_pos + normalized_direction
        
        is_nominal_safe = self.evaluate_ncbf_safety_all_objects(next_pos)
        
        if is_nominal_safe:
            return normalized_direction, True
        
        candidate_actions = []
        
        angle_step = 2 * np.pi / num_samples
        for i in range(num_samples):
            angle = i * angle_step
            rotation = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)]
            ])
            rotated_direction = np.zeros(3)
            rotated_direction[:2] = rotation @ normalized_direction[:2]
            candidate_actions.append(rotated_direction)
        
        for scale in [0.25, 0.5, 0.75]:
            scaled_action = normalized_direction * scale
            candidate_actions.append(scaled_action)
        
        for action in candidate_actions:
            next_pos = current_pos + action
            is_safe = self.evaluate_ncbf_safety_all_objects(next_pos)
            
            if is_safe:
                return action, True
        
        self.ncbf_preventions += 1
        self.get_logger().warn('No safe action found (considering all objects), staying still')
        return np.zeros(3), False

        # ===== Force retreat when no safe action found =====
        # self.ncbf_preventions += 1
        # self.get_logger().warn('No safe action found (considering all objects), FORCING step backward')
        
        # # Force retreat action (opposite direction of nominal) - NO SAFETY CHECK
        # retreat_direction = -normalized_direction
        # return retreat_direction, True  # Return True to indicate action should be executed

    
    def execute_action(self, action):
        """Execute action using frankx's PathMotion"""
        global frankx_robot
        if frankx_robot is None:
            self.get_logger().error("Cannot execute action, robot not connected")
            return False
        
        try:
            current_pose = frankx_robot.current_pose()
            current_trans = current_pose.translation()
            
            new_trans = [
                current_trans[0] + action[0],
                current_trans[1] + action[1],
                current_trans[2] + action[2]
            ]

            new_trans[2] = 0.2667 # TODO: REMOVE FIX Z 0.146, 0.2458
            
            next_pose = Affine(
                new_trans[0], new_trans[1], new_trans[2],
                current_pose.a, current_pose.b, current_pose.c
            )

            
            
            waypoints = [next_pose]
            motion = PathMotion(waypoints, blend_max_distance=0.01)
            
            frankx_robot.move(motion)
            
            self.get_logger().info(f'Executing action: [{action[0]:.4f}, {action[1]:.4f}, {action[2]:.4f}]')
            return True
            
        except Exception as e:
            self.get_logger().error(f'Failed to execute action: {str(e)}')
            import traceback
            traceback.print_exc()
            return False
    
    def get_all_object_positions(self):
        """Get positions of all valid objects"""
        positions = {}
        for tag_id in self.tag_ids:
            if self.is_tag_data_valid(tag_id):
                history = self.objects[tag_id]['history']
                if len(history) > 0:
                    positions[tag_id] = history[-1][1:4]
        return positions
    
    def run_control_loop(self, max_steps=500):
        """Main control loop"""
        self.total_steps = 0
        reached_target = False
        
        # Start recording trajectory
        self.data_recorder.start_trajectory()
        
        self.get_logger().info(f'Waiting for AprilTag data (primary tag: {self.primary_tag_id})...')
        wait_count = 0
        while not self.is_tag_data_valid(self.primary_tag_id) and wait_count < 50:
            time.sleep(0.1)
            wait_count += 1
        
        if not self.is_tag_data_valid(self.primary_tag_id):
            self.get_logger().error(f'Timeout waiting for primary tag {self.primary_tag_id} data')
            return {
                'total_steps': 0,
                'reached_target': False,
                'safety_violations': 0,
                'ncbf_preventions': 0,
                'trajectory_file': None
            }
        
        valid_tags = self.get_valid_tags()
        self.get_logger().info(f'AprilTag data received for tags: {valid_tags}')
        self.get_logger().info('Starting control loop with data recording enabled')
        
        while self.total_steps < max_steps and not reached_target:
            if not self.is_tag_data_valid(self.primary_tag_id):
                self.get_logger().warn(f'Waiting for valid data from primary tag {self.primary_tag_id}...')
                time.sleep(0.5)
                continue
            
            ee_x, ee_y, ee_z = self.get_ee_position()
            if ee_x is None:
                self.get_logger().warn('Cannot get EE position')
                time.sleep(1)
                continue
            
            current_pos = np.array([ee_x, ee_y, ee_z])
            self.ee_position = current_pos
            
            distance = np.linalg.norm(current_pos[:2] - self.target_point[:2])
            
            if distance <= self.goal_threshold:
                self.get_logger().info(f'Target reached! Distance: {distance:.4f}')
                reached_target = True
                break
            
            action, is_safe = self.select_safe_action(current_pos, self.target_point)
            
            
            if np.linalg.norm(action) > 0.0001:
                success = self.execute_action(action)
                if not success:
                    self.get_logger().warn("Action execution failed, pausing for 1 second")
                    time.sleep(1)
            else:
                time.sleep(0.1)
            
            # Record data after action execution
            self.data_recorder.record_timestep(self)
            
            for tag_id in self.get_valid_tags():
                history = self.objects[tag_id]['history']
                if len(history) > 0:
                    current_tilt = history[-1][0]
                    tilt_degrees = np.degrees(current_tilt)
                    
                    if tilt_degrees > self.max_tipping_angle:
                        self.safety_violations += 1
                        self.get_logger().warn(f'Safety violation on tag {tag_id}! Tilt angle: {tilt_degrees:.2f}°')
            
            self.total_steps += 1
            
            if self.total_steps % 5 == 0:
                self.get_logger().info(f'Step: {self.total_steps}, Distance to target: {distance:.4f}m')
                self.get_logger().info(f'EE position: [{ee_x:.4f}, {ee_y:.4f}, {ee_z:.4f}]')
                
                # Print detailed status for each tracked object
                for tag_id in self.get_valid_tags():
                    history = self.objects[tag_id]['history']
                    if len(history) > 0:
                        # Extract tilt angle
                        current_tilt = history[-1][0]
                        tilt_degrees = np.degrees(current_tilt)
                        
                        # Extract position
                        pos = history[-1][1:4]
                        
                        # Calculate CBF value at current position
                        is_safe, cbf_value = self.evaluate_ncbf_safety(
                            current_pos, tag_id, return_value=True  # ← 使用新参数
                        )
                        
                        # Print all info in one line
                        self.get_logger().info(
                            f'Tag {tag_id} - Pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}], '
                            f'Tilt: {tilt_degrees:.2f}°, CBF: {cbf_value:.4f}, Safe: {is_safe}'
                        )
            
            time.sleep(0.1)
        
        # Save trajectory
        trajectory_file = self.data_recorder.save_trajectory()
        
        # Print trajectory statistics
        stats = self.data_recorder.get_trajectory_stats()
        if stats:
            self.get_logger().info("\n=== Trajectory Statistics ===")
            self.get_logger().info(f"Total timesteps: {stats['total_steps']}")
            self.get_logger().info(f"Number of objects: {stats['num_objects']}")
            self.get_logger().info(f"Safe timesteps: {stats['safe_steps']}")
            self.get_logger().info(f"Unsafe timesteps: {stats['unsafe_steps']}")
        
        return {
            'total_steps': self.total_steps,
            'reached_target': reached_target,
            'safety_violations': self.safety_violations,
            'ncbf_preventions': self.ncbf_preventions,
            'trajectory_file': trajectory_file
        }


def ros_spin_thread(node):
    """Thread function to spin ROS2 node"""
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f'ROS spin thread error: {e}')


def signal_handler(sig, frame):
    """Handle program termination signal"""
    print('Program terminating, cleaning up resources...')
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser(description='NCBF-guided Frankx control system with ROS2 integration and data recording')
    parser.add_argument('--host', default='192.168.1.116', help='IP address of Franka robot')
    parser.add_argument('--max_steps', type=int, default=200, help='Maximum number of steps')
    parser.add_argument('--velocity', type=float, default=0.01, help='Robot movement velocity')
    parser.add_argument('--tag_ids', type=int, nargs='+', default=[2], help='List of AprilTag IDs to track')
    parser.add_argument('--primary_tag', type=int, default=None, help='Primary tag ID for control')
    parser.add_argument('--output_dir', type=str, default='real_robot_trajectories', help='Directory to save trajectory data')
    
    args = parser.parse_args()
    
    rclpy.init()
    
    signal.signal(signal.SIGINT, signal_handler)
    
    ncbf_system = SimplifiedNCBFSystem(
        robot_ip=args.host,
        tag_ids=args.tag_ids
    )
    
    # Set output directory for data recorder
    ncbf_system.data_recorder.output_dir = args.output_dir
    
    if args.primary_tag is not None:
        if args.primary_tag in args.tag_ids:
            ncbf_system.primary_tag_id = args.primary_tag
        else:
            ncbf_system.get_logger().warn(f'Primary tag {args.primary_tag} not in tag_ids, using default')
    
    ncbf_system.current_velocity = args.velocity
    
    spin_thread = threading.Thread(target=ros_spin_thread, args=(ncbf_system,), daemon=True)
    spin_thread.start()
    
    ncbf_system.get_logger().info("=== NCBF-guided Frankx control system started (with data recording) ===")
    ncbf_system.get_logger().info(f"Robot IP: {args.host}")
    ncbf_system.get_logger().info(f"Maximum steps: {args.max_steps}")
    ncbf_system.get_logger().info(f"Movement velocity: {args.velocity}")
    ncbf_system.get_logger().info(f"Tracking tag IDs: {args.tag_ids}")
    ncbf_system.get_logger().info(f"Primary tag: {ncbf_system.primary_tag_id}")
    ncbf_system.get_logger().info(f"Output directory: {args.output_dir}")
    
    try:
        results = ncbf_system.run_control_loop(max_steps=args.max_steps)
        
        ncbf_system.get_logger().info("\n=== Run Results ===")
        ncbf_system.get_logger().info(f"Total steps: {results['total_steps']}")
        ncbf_system.get_logger().info(f"Target reached: {'Yes' if results['reached_target'] else 'No'}")
        ncbf_system.get_logger().info(f"Safety violations: {results['safety_violations']}")
        ncbf_system.get_logger().info(f"NCBF preventions: {results['ncbf_preventions']}")
        if results['trajectory_file']:
            ncbf_system.get_logger().info(f"Trajectory saved to: {results['trajectory_file']}")
        
    except KeyboardInterrupt:
        ncbf_system.get_logger().info("\nProgram interrupted by user")
    except Exception as e:
        ncbf_system.get_logger().error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ncbf_system.get_logger().info("Shutting down...")
        ncbf_system.destroy_node()
        rclpy.shutdown()
        print("Program exited")


if __name__ == '__main__':
    main()