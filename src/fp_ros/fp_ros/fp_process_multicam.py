import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
from fp_msg.msg import SyncedPairs
from collections import deque
import numpy as np
import cv2
import sys
import os

from .foundationpose.estimater import *
from .foundationpose.datareader import *
import argparse

class MultiCameraPoseEstimator(Node):
    def __init__(self):
        super().__init__('multi_camera_pose_estimator')

        # Declare and get parameters
        self.declare_parameter('mesh_file', '')
        self.declare_parameter('debug_dir', '')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('est_refine_iter', 5)
        self.declare_parameter('track_refine_iter', 2)
        self.declare_parameter('debug', 1)
        self.declare_parameter('buffer_size', 3)  # Set the buffer size for each camera

        self.mesh_file = self.get_parameter('mesh_file').value
        self.debug_dir = self.get_parameter('debug_dir').value
        self.output_dir = self.get_parameter('output_dir').value
        self.est_refine_iter = self.get_parameter('est_refine_iter').value
        self.track_refine_iter = self.get_parameter('track_refine_iter').value
        self.debug = self.get_parameter('debug').value
        self.buffer_size = 50

        self.cv_bridge = CvBridge()

        # Essentially we want to keep a buffer to save the async frames so the last frame
        # can always be used for processing. Change this to something that remains synced over time
        self.color_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.depth_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.mask_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.K_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.ob_in_cam_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]

        self.subscription1 = self.create_subscription(
            SyncedPairs,
            'cam1/synced_pairs',
            self.callback1,
            QoSProfile(depth=10))
        self.subscription2 = self.create_subscription(
            SyncedPairs,
            'cam2/synced_pairs',
            self.callback2,
            QoSProfile(depth=10))
        self.subscription3 = self.create_subscription(
            SyncedPairs,
            'cam3/synced_pairs',
            self.callback3,
            QoSProfile(depth=10))

        self.initialize_pose_estimation()
        self.create_timer(0.01, self.process_frames)
        self.frame_count = 0

    def callback1(self, msg):
        color = self.cv_bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="bgr8")
        depth = self.cv_bridge.imgmsg_to_cv2(msg.depth, desired_encoding="passthrough")
        mask = self.cv_bridge.imgmsg_to_cv2(msg.segmentation, desired_encoding="mono8").astype(bool)
        ob_in_cam1_transform = None # placeholder to represent the transform from object to cam1
        K = np.array(msg.camera_matrix).reshape(3, 3)

        self.color_buffers[0].append(color)
        self.depth_buffers[0].append(depth)
        self.mask_buffers[0].append(mask)
        self.K_buffers[0].append(K)
        self.ob_in_cam_buffers[0].append(ob_in_cam1_transform)

    def callback2(self, msg):
        color = self.cv_bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="bgr8")
        depth = self.cv_bridge.imgmsg_to_cv2(msg.depth, desired_encoding="passthrough")
        mask = self.cv_bridge.imgmsg_to_cv2(msg.segmentation, desired_encoding="mono8").astype(bool)
        ob_in_cam2_transform = None # placeholder to represent the transform from object to cam1        
        K = np.array(msg.camera_matrix).reshape(3, 3)

        self.color_buffers[1].append(color)
        self.depth_buffers[1].append(depth)
        self.mask_buffers[1].append(mask)
        self.K_buffers[1].append(K)
        self.ob_in_cam_buffers[0].append(ob_in_cam2_transform)

    def callback3(self, msg):
        color = self.cv_bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="bgr8")
        depth = self.cv_bridge.imgmsg_to_cv2(msg.depth, desired_encoding="passthrough")
        mask = self.cv_bridge.imgmsg_to_cv2(msg.segmentation, desired_encoding="mono8").astype(bool)
        ob_in_cam3_transform = None # placeholder to represent the transform from object to cam1        
        K = np.array(msg.camera_matrix).reshape(3, 3)

        self.color_buffers[2].append(color)
        self.depth_buffers[2].append(depth)
        self.mask_buffers[2].append(mask)
        self.K_buffers[2].append(K)
        self.ob_in_cam_buffers[0].append(ob_in_cam3_transform)


    def initialize_pose_estimation(self):
        set_logging_format()
        set_seed(0)
        self.mesh = trimesh.load(self.mesh_file)
        os.system(f'mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')
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

        # Main processing loop
        self.window_size = 3
        self.mtc_threshold = 0.7
        self.masks = deque(maxlen=self.window_size)
        self.consecutive_occlusions = 0
        self.active_camera = 'main'  # Start with the main camera            

    def has_pixels(self, mask: np.ndarray) -> bool:
        return np.sum(mask) > 20

    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            return mask.astype(bool)        
        mask1 = ensure_binary_mask(mask1)
        mask2 = ensure_binary_mask(mask2)
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def calculate_mtc(self, masks: deque, window_size: int = 3) -> float:
        window_ious = [self.calculate_iou(masks[j], masks[j+1]) 
                    for j in range(window_size-1)]
        return np.mean(window_ious)

    def detect_occlusion(self, mtc_score: float, threshold: float = 0.7) -> bool:
        return mtc_score < threshold
    

    def process_frames(self):
        if all(len(buffer) >= 1 for buffer in self.color_buffers):
            # Process the frames
            for i in range(len(self.color_buffers[0])):
                self.frame_count += 1

                color_maincam = self.color_buffers[0][i]
                depth_maincam = self.depth_buffers[0][i]
                mask_maincam = self.mask_buffers[0][i]
                K_maincam = self.K_buffers[0][i]
                ob_in_cam_maincam = self.ob_in_cam_buffers[0][i]

                color_secondcam = self.color_buffers[1][i]
                depth_secondcam = self.depth_buffers[1][i]
                mask_secondcam = self.mask_buffers[1][i]
                K_secondcam = self.K_buffers[1][i]
                ob_in_cam_secondcam = self.ob_in_cam_buffers[1][i]

                color_thirdcam = self.color_buffers[2][i]
                depth_thirdcam = self.depth_buffers[2][i]
                mask_thirdcam = self.mask_buffers[2][i]
                K_thirdcam = self.K_buffers[2][i]
                ob_in_cam_thirdcam = self.ob_in_cam_buffers[2][i]

                if active_camera == 'main':
                    self.masks.append(mask_maincam)

                    if i == 0:
                        pose_maincam = self.est.register(K=K_maincam, rgb=color_maincam, depth=depth_maincam, ob_mask=mask_maincam, iteration=self.est_refine_iter)
                    else:
                        if len(self.masks) == self.window_size:
                            mtc_score = self.calculate_mtc(self.masks, self.window_size)
                            print(f"Current mtc_score is {mtc_score}")
                            
                            if self.detect_occlusion(mtc_score, self.mtc_threshold):
                                consecutive_occlusions += 1
                                print(f"Potential occlusion detected at frame {i} with mTC score: {mtc_score}")
                                
                                if consecutive_occlusions >= 1:
                                    print("Occlusion detected in 3 consecutive frames. Switching camera.")
                                    # Decide which camera to switch to
                                    if self.has_pixels(mask_secondcam):
                                        active_camera = 'second'
                                        pose_active = ob_in_cam_secondcam
                                        self.est.pose_last = torch.from_numpy(pose_active.reshape(1, 4, 4)) # used for track_one calculations but not shown here
                                    elif self.has_pixels(mask_thirdcam):
                                        active_camera = 'third'
                                        pose_active = ob_in_cam_thirdcam
                                        self.est.pose_last = torch.from_numpy(pose_active.reshape(1, 4, 4))
                                    else:
                                        print("No valid alternative camera found. Stopping execution.")
                                        break
                                    self.masks.clear()  # Clear the masks deque for the new camera
                                    consecutive_occlusions = 0
                            else:
                                consecutive_occlusions = 0

                        if active_camera == 'main':  # Only track if we're still using the main camera
                            pose_maincam = self.est.track_one(rgb=color_maincam, depth=depth_maincam, mask=mask_maincam, K=K_maincam, iteration=self.track_refine_iter)
                if active_camera != 'main':
                    # Use the active secondary cam
                    if active_camera == 'second':
                        color_active = color_secondcam
                        depth_active = depth_secondcam
                        mask_active = mask_secondcam
                        ob_in_cam_active = ob_in_cam_secondcam
                        K_active = K_secondcam
                    else:  # third cam
                        color_active = color_thirdcam
                        depth_active = depth_thirdcam
                        mask_active = mask_thirdcam
                        ob_in_cam_active = ob_in_cam_thirdcam
                        K_active = K_thirdcam

                    self.masks.append(mask_active)
                    pose_active = self.est.track_one(rgb=color_active, depth=depth_active, mask=mask_active, K=K_active, iteration=self.track_refine_iter)

                [os.makedirs(f'{self.debug_dir}/ob_in_cam/{cam_name}', exist_ok=True) for cam_name in ['main', 'second', 'third']]
                if active_camera == 'main':
                    np.savetxt(f'{self.debug_dir}/ob_in_cam/{active_camera}/{i}.txt', pose_maincam.reshape(4,4))
                else:
                    np.savetxt(f'{self.debug_dir}/ob_in_cam/{active_camera}/{i}.txt', pose_active.reshape(4,4))

                if self.debug >= 1:
                    [os.makedirs(os.path.join(self.output_dir, cam_name), exist_ok=True) for cam_name in ['main', 'second', 'third']]

                    if active_camera == 'main':
                        color = color_maincam
                        K = K_maincam
                        pose = pose_maincam
                    elif active_camera == 'second':
                        color = color_secondcam
                        K = K_secondcam
                        pose = pose_active
                    else: 
                        color = color_thirdcam
                        K = K_thirdcam
                        pose = pose_active

                    center_pose = pose @ np.linalg.inv(self.to_origin)
                    vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)            
                    cv2.imwrite(os.path.join(self.output_dir, active_camera, f'frame_{i:04d}.png'), vis[...,::-1])

                    print(f"Frame {i} saved in {active_camera} camera directory")


                os.makedirs(f'{self.debug_dir}/ob_in_cam', exist_ok=True)
                np.savetxt(f'{self.debug_dir}/ob_in_cam/frame_{self.frame_count:04d}.txt', pose.reshape(4,4))

                if self.debug >= 1:
                    center_pose = pose @ np.linalg.inv(self.to_origin)
                    vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
                    vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
                    cv2.imwrite(os.path.join(self.output_dir, f'frame_{self.frame_count:04d}.png'), vis[...,::-1])

def main(args=None):
    rclpy.init(args=args)
    multi_camera_pose_estimator = MultiCameraPoseEstimator()
    rclpy.spin(multi_camera_pose_estimator)
    multi_camera_pose_estimator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()