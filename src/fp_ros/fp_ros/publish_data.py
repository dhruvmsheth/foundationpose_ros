import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
import cv2
import os
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from fp_msg.msg import SyncedPairs
import time

class ImagePublisher(Node):
    def __init__(self):
        super().__init__('image_publisher')
        
        self.publisher = self.create_publisher(SyncedPairs, 'cam1/synced_pairs', QoSProfile(depth=10))
        self.timer = self.create_timer(1/5, self.publish_images)  # 30Hz
        
        self.cv_bridge = CvBridge()
        self.data_path = '/home/FoundationPose_ros/src/fp_ros/fp_ros/data/demo_data/unity_sim_data'
        self.rgb_path = os.path.join(self.data_path, 'rgb')
        self.depth_path = os.path.join(self.data_path, 'depth')
        self.mask_path = os.path.join(self.data_path, 'masks')
        
        self.image_files = sorted(os.listdir(self.rgb_path))
        self.current_index = 0
        self.K = np.loadtxt('/home/FoundationPose_ros/src/fp_ros/fp_ros/data/demo_data/unity_sim_data/cam_K.txt').reshape(3,3)

    def publish_images(self):
        if self.current_index >= len(self.image_files):
            self.current_index = 0

        filename = self.image_files[self.current_index]
        
        # Read RGB image
        rgb_img = cv2.imread(os.path.join(self.rgb_path, filename))
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8')        
        depth_img = cv2.imread(os.path.join(self.depth_path, filename), cv2.IMREAD_ANYDEPTH)
        depth_img_float = depth_img.astype(np.float32)        
        depth_img_float /= 1000.0  
        depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_img_float, encoding='32FC1')        
        seg_img = cv2.imread(os.path.join(self.mask_path, filename), cv2.IMREAD_GRAYSCALE)
        seg_msg = self.cv_bridge.cv2_to_imgmsg(seg_img, encoding='mono8')
        # Create SyncedPairs message
        synced_pairs_msg = SyncedPairs()
        synced_pairs_msg.header = Header()
        synced_pairs_msg.header.stamp = self.get_clock().now().to_msg()
        synced_pairs_msg.header.frame_id = 'camera_frame'
        synced_pairs_msg.rgb = rgb_msg
        synced_pairs_msg.depth = depth_msg
        synced_pairs_msg.segmentation = seg_msg
        synced_pairs_msg.camera_matrix = self.K.flatten().tolist()
        self.publisher.publish(synced_pairs_msg)
        
        self.current_index += 1
        
        self.get_logger().info(f'Published image set {self.current_index}/{len(self.image_files)}')

    # def publish_images(self):
    #     if self.current_index >= len(self.image_files):
    #         self.current_index = 0
    #     filename = self.image_files[self.current_index]        
    #     rgb_img = cv2.imread(os.path.join(self.rgb_path, filename))
    #     rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8')
    #     depth_img = cv2.imread(os.path.join(self.depth_path, filename), cv2.IMREAD_ANYDEPTH)
    #     depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_img, encoding='32FC1')
    #     # Read segmentation image and convert to mono8

    #     seg_img = cv2.imread(os.path.join(self.mask_path, filename), cv2.IMREAD_GRAYSCALE)
    #     seg_msg = self.cv_bridge.cv2_to_imgmsg(seg_img, encoding='mono8')

    #     synced_pairs_msg = SyncedPairs()
    #     synced_pairs_msg.header = Header()
    #     synced_pairs_msg.header.stamp = self.get_clock().now().to_msg()
    #     synced_pairs_msg.header.frame_id = 'camera_frame'
    #     synced_pairs_msg.rgb = rgb_msg
    #     synced_pairs_msg.depth = depth_msg
    #     synced_pairs_msg.segmentation = seg_msg
    #     synced_pairs_msg.camera_matrix = self.K.flatten().tolist()  # Add camera matrix
        
    #     self.publisher.publish(synced_pairs_msg)
    #     self.current_index += 1
        
    #     self.get_logger().info(f'Published image set {self.current_index}/{len(self.image_files)}')

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()