import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
from fp_msg.msg import SyncedPairs
from queue import Queue
import numpy as np
import cv2
import sys
import os

# print(f"Python path: {sys.path}")
# print(f"Current working directory: {os.getcwd()}")
# print(f"Contents of current directory: {os.listdir()}")
# print(f"Contents of fp_ros: {os.listdir('/home/FoundationPose_ros/install/fp_ros/lib/python3.10/site-packages/fp_ros')}")
# print(f"Contents of foundationpose: {os.listdir('/home/FoundationPose_ros/install/fp_ros/lib/python3.10/site-packages/fp_ros/foundationpose')}")

from .foundationpose.estimater import *
from .foundationpose.datareader import *
import argparse

class PoseEstimator(Node):
    def __init__(self):
        super().__init__('pose_estimator')

        # Declare and get parameters
        self.declare_parameter('mesh_file', '')
        self.declare_parameter('debug_dir', '')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('est_refine_iter', 5)
        self.declare_parameter('track_refine_iter', 2)
        self.declare_parameter('debug', 1)
        self.mesh_file = self.get_parameter('mesh_file').value
        self.debug_dir = self.get_parameter('debug_dir').value
        self.output_dir = self.get_parameter('output_dir').value
        self.est_refine_iter = self.get_parameter('est_refine_iter').value
        self.track_refine_iter = self.get_parameter('track_refine_iter').value
        self.debug = self.get_parameter('debug').value

        self.cv_bridge = CvBridge()
        self.msg_queue = Queue(maxsize=100)

        self.subscription = self.create_subscription(
            SyncedPairs,
            'cam1/synced_pairs',
            self.callback,
            QoSProfile(depth=10))

        self.initialize_pose_estimation()
        self.create_timer(0.01, self.process_queue)
        self.K = None

    def callback(self, msg):
        if self.msg_queue.full():
            self.msg_queue.get()
        self.msg_queue.put(msg)

    def initialize_pose_estimation(self):
        set_logging_format()
        set_seed(0)
        self.mesh = trimesh.load(self.mesh_file)
        os.system(f'rm -rf {self.debug_dir}/* && mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')
        os.makedirs(self.output_dir, exist_ok=True)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        self.est = FoundationPose(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, 
                                  mesh=self.mesh, scorer=scorer, refiner=refiner, debug_dir=self.debug_dir, 
                                  debug=self.debug, glctx=glctx)
        self.get_logger().info("Estimator initialization done")
        self.frame_count = 0

    def process_queue(self):
        if self.msg_queue.empty():
            return

        msg = self.msg_queue.get()
        self.frame_count += 1

        color = self.cv_bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="bgr8")
        depth = self.cv_bridge.imgmsg_to_cv2(msg.depth, desired_encoding="passthrough")
        mask = self.cv_bridge.imgmsg_to_cv2(msg.segmentation, desired_encoding="mono8").astype(bool)
        self.K = np.array(msg.camera_matrix).reshape(3, 3)

        print(f"Color dtype: {color.dtype}, shape: {color.shape}")
        print(f"Depth dtype: {depth.dtype}, shape: {depth.shape}")
        print(f"Mask dtype: {mask.dtype}, shape: {mask.shape}")

        # Convert depth to float32
        # depth = depth.astype(np.float32)        

        if self.frame_count == 1:
            pose = self.est.register(K=self.K, rgb=color, depth=depth, ob_mask=mask, iteration=self.est_refine_iter)
            if self.debug >= 3:
                m = self.mesh.copy()
                m.apply_transform(pose)
                m.export(f'{self.debug_dir}/model_tf.obj')
                xyz_map = depth2xyzmap(depth, self.K)
                valid = depth >= 0.1
                pcd = toOpen3dCloud(xyz_map[valid], color[valid])
                o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply', pcd)
        else:
            pose = self.est.track_one(rgb=color, depth=depth, mask=mask, K=self.K, iteration=self.track_refine_iter)

        os.makedirs(f'{self.debug_dir}/ob_in_cam', exist_ok=True)
        np.savetxt(f'{self.debug_dir}/ob_in_cam/frame_{self.frame_count:04d}.txt', pose.reshape(4,4))

        if self.debug >= 1:
            center_pose = pose @ np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
            cv2.imwrite(os.path.join(self.output_dir, f'frame_{self.frame_count:04d}.png'), vis[...,::-1])

        if self.debug >= 2:
            os.makedirs(f'{self.debug_dir}/track_vis', exist_ok=True)
            imageio.imwrite(f'{self.debug_dir}/track_vis/frame_{self.frame_count:04d}.png', vis)

def main(args=None):
    rclpy.init(args=args)
    pose_estimator = PoseEstimator()
    rclpy.spin(pose_estimator)
    pose_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()