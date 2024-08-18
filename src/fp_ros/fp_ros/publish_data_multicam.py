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

        # Renamed publishers to publishers_dict to avoid attribute conflict
        self.publishers_dict = {
            'cam1': self.create_publisher(SyncedPairs, 'cam1/synced_pairs', QoSProfile(depth=10)),
            'cam2': self.create_publisher(SyncedPairs, 'cam2/synced_pairs', QoSProfile(depth=10)),
            'cam3': self.create_publisher(SyncedPairs, 'cam3/synced_pairs', QoSProfile(depth=10))
        }

        self.timer = self.create_timer(1/5, self.publish_images)  # 5Hz

        self.cv_bridge = CvBridge()
        self.data_path = '/home/FoundationPose_ros/src/fp_ros/fp_ros/data/demo_data/img'
        
        self.camera_data = {
            'cam1': self.load_camera_data('cam1'),
            'cam2': self.load_camera_data('cam2'),
            'cam3': self.load_camera_data('cam3')
        }
        
        self.current_index = 0

    def load_camera_data(self, cam_name):
        cam_path = os.path.join(self.data_path, cam_name)
        rgb_path = os.path.join(cam_path, 'rgb')
        depth_path = os.path.join(cam_path, 'depth')
        mask_path = os.path.join(cam_path, 'mask')
        K = np.loadtxt(os.path.join(cam_path, 'K.txt')).reshape(3, 3)
        
        image_files = sorted(os.listdir(rgb_path))

        return {
            'rgb_path': rgb_path,
            'depth_path': depth_path,
            'mask_path': mask_path,
            'K': K,
            'image_files': image_files
        }

    def publish_images(self):
        for cam_name, cam_data in self.camera_data.items():
            if self.current_index >= len(cam_data['image_files']):
                self.current_index = 0
            
            filename = cam_data['image_files'][self.current_index]

            # Read RGB image
            rgb_img = cv2.imread(os.path.join(cam_data['rgb_path'], filename))
            rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
            rgb_msg = self.cv_bridge.cv2_to_imgmsg(rgb_img, encoding='bgr8')
            
            # Read Depth image
            depth_img = cv2.imread(os.path.join(cam_data['depth_path'], filename), cv2.IMREAD_ANYDEPTH)
            depth_img_float = depth_img.astype(np.float32)        
            depth_img_float /= 1000.0  
            depth_msg = self.cv_bridge.cv2_to_imgmsg(depth_img_float, encoding='32FC1')
            
            # Read Mask image
            seg_img = cv2.imread(os.path.join(cam_data['mask_path'], filename), cv2.IMREAD_GRAYSCALE)
            seg_msg = self.cv_bridge.cv2_to_imgmsg(seg_img, encoding='mono8')

            # Create SyncedPairs message
            synced_pairs_msg = SyncedPairs()
            synced_pairs_msg.header = Header()
            synced_pairs_msg.header.stamp = self.get_clock().now().to_msg()
            synced_pairs_msg.header.frame_id = f'{cam_name}_frame'
            synced_pairs_msg.rgb = rgb_msg
            synced_pairs_msg.depth = depth_msg
            synced_pairs_msg.segmentation = seg_msg
            synced_pairs_msg.camera_matrix = cam_data['K'].flatten().tolist()

            self.publishers_dict[cam_name].publish(synced_pairs_msg)
            
        self.current_index += 1
        
        self.get_logger().info(f'Published image set {self.current_index}/{len(self.camera_data["cam1"]["image_files"])}')

def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher()
    rclpy.spin(image_publisher)
    image_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
