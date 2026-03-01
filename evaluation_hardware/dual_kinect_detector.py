#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray, Quaternion, Point, Pose
from cv_bridge import CvBridge
import cv2
import numpy as np
from datetime import datetime
import os

class MultiAprilTagDetector(Node):
    def __init__(self):
        super().__init__('apriltag_detector')
        
        # Declare parameters
        self.declare_parameter('camera_name', 'kinect1')
        self.declare_parameter('frame_id', 'camera1_color_optical_frame')
        self.declare_parameter('output_dir', './videos')  # Video output directory
        
        # Get parameters
        self.camera_name = self.get_parameter('camera_name').value
        self.frame_id = self.get_parameter('frame_id').value
        self.output_dir = self.get_parameter('output_dir').value
    
        # Create resizable window (MOVED HERE)
        cv2.namedWindow(f'AprilTag Detection - {self.camera_name}', cv2.WINDOW_NORMAL)
          
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # Video writer
        self.video_writer = None
        self.video_filename = None
        
        # AprilTag detector setup (OpenCV 4.7+)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # AprilTag actual size (unit: meters)
        self.marker_length = 0.04  # Modify to your AprilTag's actual side length
        
        # Create publishers with camera-specific prefix
        self.pose_pubs = {}  # Dictionary to store publishers for each tag ID
        self.pose_topic_prefix = f'/{self.camera_name}/apriltag'
        
        # Publish all poses together in a PoseArray
        self.pose_array_pub = self.create_publisher(
            PoseArray, 
            f'{self.pose_topic_prefix}/pose_array', 
            10
        )
        
        # Subscribe to RGB image
        self.image_sub = self.create_subscription(
            Image,
            f'{self.camera_name}/rgb/image_raw',
            self.image_callback,
            10
        )
        
        # Subscribe to camera info
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            f'{self.camera_name}/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.get_logger().info(f'AprilTag detector started for {self.camera_name}')
        self.get_logger().info(f'Frame ID: {self.frame_id}')
        self.get_logger().info(f'Publishing to: {self.pose_topic_prefix}/tag_{{id}}/pose')
        self.get_logger().info(f'OpenCV version: {cv2.__version__}')
        self.get_logger().info(f'Video output directory: {self.output_dir}')
    
    def __del__(self):
        """Destructor to release video writer"""
        if self.video_writer is not None:
            self.video_writer.release()
            self.get_logger().info(f'Video saved: {self.video_filename}')
    
    def camera_info_callback(self, msg):
        """Get camera calibration parameters"""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info(f'{self.camera_name}: Camera calibration info received')
    
    def initialize_video_writer(self, frame_shape):
        """Initialize video writer with frame dimensions"""
        if self.video_writer is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            self.video_filename = os.path.join(
                self.output_dir, 
                f'apriltag_detection_{self.camera_name}_{timestamp}.mp4'
            )
            
            height, width = frame_shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = 30.0  # Adjust based on your camera's actual frame rate
            
            self.video_writer = cv2.VideoWriter(
                self.video_filename,
                fourcc,
                fps,
                (width, height)
            )
            
            self.get_logger().info(f'Video writer initialized: {self.video_filename}')
    
    def image_callback(self, msg):
        """Process image and publish all AprilTag poses"""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Initialize video writer on first frame
            if self.video_writer is None:
                self.initialize_video_writer(cv_image.shape)
            
            # Exit if no camera calibration info available
            if self.camera_matrix is None:
                return
            
            # Detect AprilTag (using new API)
            corners, ids, rejected = self.detector.detectMarkers(cv_image)
            
            # If markers detected
            if ids is not None and len(ids) > 0:
                # Estimate pose for each marker
                all_poses = []
                
                for i, corner in enumerate(corners):
                    tag_id = ids[i][0]
                    
                    # Define 3D points of the marker corners in marker coordinate system
                    obj_points = np.array([
                        [-self.marker_length/2,  self.marker_length/2, 0],
                        [ self.marker_length/2,  self.marker_length/2, 0],
                        [ self.marker_length/2, -self.marker_length/2, 0],
                        [-self.marker_length/2, -self.marker_length/2, 0]
                    ], dtype=np.float32)
                    
                    # Solve PnP to get rotation and translation vectors
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        corner[0],
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )
                    
                    if success:
                        tvec = tvec.flatten()
                        rvec = rvec.flatten()
                        
                        # Draw detection results
                        cv2.aruco.drawDetectedMarkers(cv_image, [corner], np.array([[tag_id]]))
                        cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, 
                                        rvec, tvec, self.marker_length * 0.5)
                        
                        # Convert rotation vector to quaternion
                        rot_matrix, _ = cv2.Rodrigues(rvec)
                        quat = self._rotation_matrix_to_quaternion(rot_matrix)
                        
                        # Create pose message for this marker
                        pose_msg = PoseStamped()
                        pose_msg.header.frame_id = self.frame_id  # Use configured frame_id
                        pose_msg.header.stamp = self.get_clock().now().to_msg()
                        pose_msg.pose.position = Point(x=float(tvec[0]), y=float(tvec[1]), z=float(tvec[2]))
                        pose_msg.pose.orientation = Quaternion(x=float(quat[0]), y=float(quat[1]), 
                                                               z=float(quat[2]), w=float(quat[3]))
                        
                        # Publish individual pose for this tag
                        topic_name = f'{self.pose_topic_prefix}/tag_{tag_id}/pose'
                        if tag_id not in self.pose_pubs:
                            self.pose_pubs[tag_id] = self.create_publisher(PoseStamped, topic_name, 10)
                            self.get_logger().info(f'Created publisher for tag ID {tag_id}: {topic_name}')
                        
                        self.pose_pubs[tag_id].publish(pose_msg)
                        
                        # Add to list for PoseArray
                        all_poses.append(pose_msg.pose)
                        
                        # Print info
                        self.get_logger().info(
                            f'{self.camera_name} - Tag ID {tag_id}: pos=[{tvec[0]:.3f}, {tvec[1]:.3f}, {tvec[2]:.3f}]',
                            throttle_duration_sec=1.0  # Only log once per second
                        )
                
                # Publish PoseArray with all detected markers
                if len(all_poses) > 0:
                    pose_array_msg = PoseArray()
                    pose_array_msg.header.frame_id = self.frame_id
                    pose_array_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_array_msg.poses = all_poses
                    self.pose_array_pub.publish(pose_array_msg)
            
            # Write frame to video
            if self.video_writer is not None:
                self.video_writer.write(cv_image)
            
            # Display results
            cv2.imshow(f'AprilTag Detection - {self.camera_name}', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Image processing error: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert rotation matrix to quaternion"""
        m00, m01, m02 = R[0, 0], R[0, 1], R[0, 2]
        m10, m11, m12 = R[1, 0], R[1, 1], R[1, 2]
        m20, m21, m22 = R[2, 0], R[2, 1], R[2, 2]
        
        trace = m00 + m11 + m22
        
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (m21 - m12) * s
            y = (m02 - m20) * s
            z = (m10 - m01) * s
        elif m00 > m11 and m00 > m22:
            s = 2.0 * np.sqrt(1.0 + m00 - m11 - m22)
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = 2.0 * np.sqrt(1.0 + m11 - m00 - m22)
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m22 - m00 - m11)
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        
        return [x, y, z, w]

def main(args=None):
    rclpy.init(args=args)
    node = MultiAprilTagDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Release video writer before destroying node
        if node.video_writer is not None:
            node.video_writer.release()
            node.get_logger().info(f'Video saved: {node.video_filename}')
        
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()