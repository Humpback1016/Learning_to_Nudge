#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray, TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
import cv2
import numpy as np

class MultiAprilTagDetector(Node):
    def __init__(self):
        super().__init__('apriltag_detector')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # TF Broadcaster 
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Camera parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        
        # AprilTag detector setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        
        # AprilTag actual size (meters)
        self.marker_length = 0.04
        
        # select a main tag ID for calibration (modify according to the actual ID used)
        self.calibration_tag_id = 1  
        
        # Publishers
        self.pose_pubs = {}
        self.pose_array_pub = self.create_publisher(PoseArray, '/apriltag/pose_array', 10)
        
        # Subscriptions
        self.image_sub = self.create_subscription(
            Image,
            'kinect1/rgb/image_raw',
            self.image_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            'kinect1/rgb/camera_info',
            self.camera_info_callback,
            10
        )
        
        self.get_logger().info('Multi AprilTag detector started')
    
    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3)
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info('Camera calibration info received')
    
    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            if self.camera_matrix is None:
                return
            
            corners, ids, rejected = self.detector.detectMarkers(cv_image)
            
            if ids is not None and len(ids) > 0:
                all_poses = []
                
                for i, corner in enumerate(corners):
                    tag_id = ids[i][0]
                    
                    obj_points = np.array([
                        [-self.marker_length/2,  self.marker_length/2, 0],
                        [ self.marker_length/2,  self.marker_length/2, 0],
                        [ self.marker_length/2, -self.marker_length/2, 0],
                        [-self.marker_length/2, -self.marker_length/2, 0]
                    ], dtype=np.float32)
                    
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
                        
                        cv2.aruco.drawDetectedMarkers(cv_image, [corner], np.array([[tag_id]]))
                        cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, 
                                        rvec, tvec, self.marker_length * 0.5)
                        
                        rot_matrix, _ = cv2.Rodrigues(rvec)
                        quat = self._rotation_matrix_to_quaternion(rot_matrix)
                        
                        # publish TF Transform
                        t = TransformStamped()
                        t.header.stamp = self.get_clock().now().to_msg()
                        t.header.frame_id = "camera_color_optical_frame"
                        t.child_frame_id = f"apriltag_{tag_id}"
                        
                        t.transform.translation.x = float(tvec[0])
                        t.transform.translation.y = float(tvec[1])
                        t.transform.translation.z = float(tvec[2])
                        
                        t.transform.rotation.x = float(quat[0])
                        t.transform.rotation.y = float(quat[1])
                        t.transform.rotation.z = float(quat[2])
                        t.transform.rotation.w = float(quat[3])
                        
                        self.tf_broadcaster.sendTransform(t)
                        
                        # continue to publish PoseStamped 
                        pose_msg = PoseStamped()
                        pose_msg.header = t.header
                        pose_msg.pose.position.x = t.transform.translation.x
                        pose_msg.pose.position.y = t.transform.translation.y
                        pose_msg.pose.position.z = t.transform.translation.z
                        pose_msg.pose.orientation = t.transform.rotation
                        
                        topic_name = f'/apriltag/tag_{tag_id}/pose'
                        if tag_id not in self.pose_pubs:
                            self.pose_pubs[tag_id] = self.create_publisher(PoseStamped, topic_name, 10)
                        
                        self.pose_pubs[tag_id].publish(pose_msg)
                        all_poses.append(pose_msg.pose)
                
                if len(all_poses) > 0:
                    pose_array_msg = PoseArray()
                    pose_array_msg.header.frame_id = "camera_color_optical_frame"
                    pose_array_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_array_msg.poses = all_poses
                    self.pose_array_pub.publish(pose_array_msg)
                
                cv2.imshow('AprilTag Detection', cv_image)
                cv2.waitKey(1)
            
        except Exception as e:
            self.get_logger().error(f'Error: {e}')
    
    def _rotation_matrix_to_quaternion(self, R):
    
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
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    main()