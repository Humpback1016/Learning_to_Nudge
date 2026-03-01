from argparse import ArgumentParser
from time import sleep
import numpy as np

from frankx import Affine, Robot

import rclpy
from rclpy.node import Node
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation


class FrankaTFPublisher(Node):
    def __init__(self, robot_ip):
        super().__init__('franka_tf_publisher')
        
        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
        
        # Connect to robot
        self.robot = Robot(robot_ip)
        self.robot.set_default_behavior()
        
        # Create timer for publishing (100 Hz recommended for TF)
        self.timer = self.create_timer(0.01, self.publish_tf)
        
        self.get_logger().info(f'Connected to Franka at {robot_ip}')
    
    def publish_tf(self):
        # Read robot state
        state = self.robot.read_once()
        
        # Franka uses COLUMN-MAJOR storage, so transpose
        matrix_4x4 = np.array(state.O_T_EE).reshape(4, 4).T
        
        # Extract translation (last column, first 3 rows)
        translation = matrix_4x4[:3, 3]
        
        # Extract rotation matrix
        rotation_matrix = matrix_4x4[:3, :3]
        r = Rotation.from_matrix(rotation_matrix)
        quaternion = r.as_quat()  # Returns [x, y, z, w]
        
        # Create transform message
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'panda_link0'
        t.child_frame_id = 'panda_EE'
        
        # Set translation
        t.transform.translation.x = float(translation[0])
        t.transform.translation.y = float(translation[1])
        t.transform.translation.z = float(translation[2])
        
        # Set rotation (quaternion)
        t.transform.rotation.x = float(quaternion[0])
        t.transform.rotation.y = float(quaternion[1])
        t.transform.rotation.z = float(quaternion[2])
        t.transform.rotation.w = float(quaternion[3])
        
        # Broadcast transform
        self.tf_broadcaster.sendTransform(t)
        

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--host', default='192.168.1.116', help='FCI IP of the robot')
    args = parser.parse_args()
    
    rclpy.init()
    
    node = FrankaTFPublisher(args.host)
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()