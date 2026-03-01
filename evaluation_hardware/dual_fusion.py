#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
import numpy as np
from collections import defaultdict
import time

class AprilTagFusionNode(Node):
    def __init__(self):
        super().__init__('apriltag_fusion')
        
        # Declare parameters
        self.declare_parameter('tag_ids', [2])
        self.declare_parameter('data_timeout', 2.0)
        self.declare_parameter('quality_hysteresis', 0.15)
        
        # Get parameters
        self.tag_ids = self.get_parameter('tag_ids').value
        self.data_timeout = self.get_parameter('data_timeout').value
        self.quality_hysteresis = self.get_parameter('quality_hysteresis').value
        
        # Hand-eye calibration results
        # TODO: Replace with your actual calibration values!
        self.camera1_to_base = {
            'translation': np.array([0.5815, 0.2578, 0.4787]),
            'rotation': np.array([-0.3878, -0.8530, 0.3221, 0.1348])  # [qx, qy, qz, qw]
        }
        
        self.camera2_to_base = {
            'translation': np.array([0.5777, -0.2472, 0.4754]),  # REPLACE WITH YOUR VALUES
            'rotation': np.array([-0.8870, -0.2131, 0.0959, 0.3982])  # REPLACE WITH YOUR VALUES
        }
        
        # Data storage for each tag
        # Structure: {tag_id: {'camera1': {...}, 'camera2': {...}, 'selected_camera': str}}
        self.tag_data = defaultdict(lambda: {
            'camera1': {'pose': None, 'timestamp': None, 'quality': 0.0},
            'camera2': {'pose': None, 'timestamp': None, 'quality': 0.0},
            'selected_camera': None
        })
        
        # Publishers for unified poses
        self.unified_publishers = {}
        
        # Subscribers for each tag from both cameras
        self.camera1_subscribers = {}
        self.camera2_subscribers = {}
        
        for tag_id in self.tag_ids:
            # Subscribe to camera1
            topic_cam1 = f'/kinect1/apriltag/tag_{tag_id}/pose'
            self.camera1_subscribers[tag_id] = self.create_subscription(
                PoseStamped,
                topic_cam1,
                lambda msg, tid=tag_id: self.camera1_callback(msg, tid),
                10
            )
            
            # Subscribe to camera2
            topic_cam2 = f'/kinect2/apriltag/tag_{tag_id}/pose'
            self.camera2_subscribers[tag_id] = self.create_subscription(
                PoseStamped,
                topic_cam2,
                lambda msg, tid=tag_id: self.camera2_callback(msg, tid),
                10
            )
            
            # Create unified publisher
            unified_topic = f'/apriltag/tag_{tag_id}/pose'
            self.unified_publishers[tag_id] = self.create_publisher(
                PoseStamped,
                unified_topic,
                10
            )
            
            self.get_logger().info(f'Tag {tag_id}: Subscribing to {topic_cam1} and {topic_cam2}')
            self.get_logger().info(f'Tag {tag_id}: Publishing unified pose to {unified_topic}')
        
        self.get_logger().info('AprilTag Fusion Node started')
        self.get_logger().info(f'Tracking tags: {self.tag_ids}')
    
    def camera1_callback(self, msg, tag_id):
        """Callback for camera1 detections"""
        # Store detection
        self.tag_data[tag_id]['camera1']['pose'] = msg
        self.tag_data[tag_id]['camera1']['timestamp'] = time.time()
        
        # Calculate quality score
        quality = self.calculate_quality(msg, tag_id, 'camera1')
        self.tag_data[tag_id]['camera1']['quality'] = quality
        
        # Trigger selection and publishing
        self.select_and_publish(tag_id)
    
    def camera2_callback(self, msg, tag_id):
        """Callback for camera2 detections"""
        # Store detection
        self.tag_data[tag_id]['camera2']['pose'] = msg
        self.tag_data[tag_id]['camera2']['timestamp'] = time.time()
        
        # Calculate quality score
        quality = self.calculate_quality(msg, tag_id, 'camera2')
        self.tag_data[tag_id]['camera2']['quality'] = quality
        
        # Trigger selection and publishing
        self.select_and_publish(tag_id)
    
    def calculate_quality(self, pose_msg, tag_id, camera_name):
        """Calculate quality score for a detection"""
        # Extract position
        pos = pose_msg.pose.position
        distance = np.sqrt(pos.x**2 + pos.y**2 + pos.z**2)
        
        # Distance score (optimal range: 0.4-0.9m)
        if 0.4 <= distance <= 0.9:
            distance_score = 1.0
        elif distance < 0.4:
            distance_score = distance / 0.4
        else:  # > 0.9
            distance_score = max(0.0, 1.0 - (distance - 0.9) / 0.6)
        
        # Viewing angle score
        quat = np.array([
            pose_msg.pose.orientation.x,
            pose_msg.pose.orientation.y,
            pose_msg.pose.orientation.z,
            pose_msg.pose.orientation.w
        ])
        angle_score = self.calculate_viewing_angle_score(quat, pos)
        
        # Freshness score
        current_time = time.time()
        timestamp = self.tag_data[tag_id][camera_name]['timestamp']
        if timestamp is not None:
            age = current_time - timestamp
            freshness_score = np.exp(-age / 1.0)  # 1 second decay
        else:
            freshness_score = 0.0
        
        # Combined score
        quality = 0.5 * distance_score + 0.3 * angle_score + 0.2 * freshness_score
        
        return quality
    
    def calculate_viewing_angle_score(self, quat, position):
        """
        Calculate viewing angle score based on how perpendicular the tag is to camera view
        Higher score = better viewing angle
        """
        # Convert quaternion to rotation matrix
        rot_matrix = self.quaternion_to_rotation_matrix(quat)
        
        # Tag's normal vector (z-axis of tag frame)
        tag_normal = rot_matrix[:, 2]
        
        # Camera view direction (towards the tag)
        # FIXED: Convert position to numpy array first
        pos_array = np.array([position.x, position.y, position.z])
        distance = np.linalg.norm(pos_array)
        
        if distance < 0.001:  # Avoid division by zero
            return 0.0
        
        view_direction = -pos_array / distance
        
        # Calculate dot product (cosine of angle)
        dot_product = np.abs(np.dot(tag_normal, view_direction))
        
        # Score: 1.0 when perpendicular, 0.0 when parallel
        angle_score = dot_product
        
        return angle_score
    
    def quaternion_to_rotation_matrix(self, q):
        """Convert quaternion to rotation matrix"""
        x, y, z, w = q
        
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        return R
    
    def is_data_valid(self, tag_id, camera_name):
        """Check if data from a camera is valid and fresh"""
        data = self.tag_data[tag_id][camera_name]
        
        if data['pose'] is None or data['timestamp'] is None:
            return False
        
        # Check freshness
        age = time.time() - data['timestamp']
        if age > self.data_timeout:
            return False
        
        return True
    
    def select_and_publish(self, tag_id):
        """Select best camera and publish unified pose"""
        cam1_valid = self.is_data_valid(tag_id, 'camera1')
        cam2_valid = self.is_data_valid(tag_id, 'camera2')
        
        if not cam1_valid and not cam2_valid:
            return  # No valid data to publish
        
        # Get current and previous selection
        previous_selection = self.tag_data[tag_id]['selected_camera']
        
        # Select best camera
        if cam1_valid and not cam2_valid:
            selected = 'camera1'
        elif cam2_valid and not cam1_valid:
            selected = 'camera2'
        else:
            # Both valid - compare quality with hysteresis
            cam1_quality = self.tag_data[tag_id]['camera1']['quality']
            cam2_quality = self.tag_data[tag_id]['camera2']['quality']
            
            # Apply hysteresis: current camera gets a small bonus
            if previous_selection == 'camera1':
                cam1_quality += self.quality_hysteresis
            elif previous_selection == 'camera2':
                cam2_quality += self.quality_hysteresis
            
            # Select camera with higher quality
            if cam1_quality > cam2_quality:
                selected = 'camera1'
            else:
                selected = 'camera2'
        
        # Log camera switch
        if previous_selection != selected and previous_selection is not None:
            self.get_logger().info(
                f'Tag {tag_id}: Switched from {previous_selection} to {selected}'
            )
        
        # Update selected camera
        self.tag_data[tag_id]['selected_camera'] = selected
        
        # Get selected pose
        selected_pose = self.tag_data[tag_id][selected]['pose']
        
        # Transform to base frame
        transformed_pose = self.transform_to_base(selected_pose, selected)
        
        if transformed_pose is not None:
            # Publish unified pose
            self.unified_publishers[tag_id].publish(transformed_pose)
            
            # Log occasionally
            if not hasattr(self, f'_log_count_{tag_id}'):
                setattr(self, f'_log_count_{tag_id}', 0)
            
            count = getattr(self, f'_log_count_{tag_id}')
            if count % 50 == 0:  # Log every 50 messages
                pos = transformed_pose.pose.position
                quality = self.tag_data[tag_id][selected]['quality']
                self.get_logger().info(
                    f'Tag {tag_id} ({selected}): pos=[{pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f}], quality={quality:.3f}'
                )
            setattr(self, f'_log_count_{tag_id}', count + 1)
    
    def transform_to_base(self, pose_stamped, camera_name):
        """Transform pose from camera frame to robot base frame"""
        try:
            # Get appropriate transform
            if camera_name == 'camera1':
                transform = self.camera1_to_base
            else:
                transform = self.camera2_to_base
            
            # Extract pose components
            pos = np.array([
                pose_stamped.pose.position.x,
                pose_stamped.pose.position.y,
                pose_stamped.pose.position.z
            ])
            
            quat = np.array([
                pose_stamped.pose.orientation.x,
                pose_stamped.pose.orientation.y,
                pose_stamped.pose.orientation.z,
                pose_stamped.pose.orientation.w
            ])
            
            # Transform position
            # Convert to homogeneous transformation matrices
            T_camera_to_base = self.pose_to_transform_matrix(
                transform['translation'], 
                transform['rotation']
            )
            T_tag_to_camera = self.pose_to_transform_matrix(pos, quat)
            
            # Chain transformations: T_tag_to_base = T_camera_to_base * T_tag_to_camera
            T_tag_to_base = T_camera_to_base @ T_tag_to_camera
            
            # Extract transformed position and orientation
            transformed_pos = T_tag_to_base[:3, 3]
            transformed_quat = self.rotation_matrix_to_quaternion(T_tag_to_base[:3, :3])
            
            # Create transformed pose message
            transformed_pose = PoseStamped()
            transformed_pose.header.frame_id = 'panda_link0'
            transformed_pose.header.stamp = pose_stamped.header.stamp
            transformed_pose.pose.position.x = float(transformed_pos[0])
            transformed_pose.pose.position.y = float(transformed_pos[1])
            transformed_pose.pose.position.z = float(transformed_pos[2])
            transformed_pose.pose.orientation.x = float(transformed_quat[0])
            transformed_pose.pose.orientation.y = float(transformed_quat[1])
            transformed_pose.pose.orientation.z = float(transformed_quat[2])
            transformed_pose.pose.orientation.w = float(transformed_quat[3])
            
            return transformed_pose
            
        except Exception as e:
            self.get_logger().error(f'Transform failed: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
            return None
    
    def pose_to_transform_matrix(self, translation, quaternion):
        """Convert position and quaternion to 4x4 transformation matrix"""
        # Extract quaternion components
        x, y, z, w = quaternion
        
        # Create rotation matrix from quaternion
        R = np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
        
        # Create 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = translation
        
        return T
    
    def rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion [x, y, z, w]"""
        trace = np.trace(R)
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        
        return np.array([x, y, z, w])

def main(args=None):
    rclpy.init(args=args)
    node = AprilTagFusionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()