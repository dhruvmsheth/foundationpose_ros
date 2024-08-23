import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from cv_bridge import CvBridge
# from fp_msg.msg import SyncedPairs
from msgs_srvs.msg import RGBDSeg

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros import TransformBroadcaster
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import TransformStamped

from queue import PriorityQueue
from collections import deque
import numpy as np
import cv2
import sys
import os
import pyquaternion as pyqt

# print(f"Python path: {sys.path}")
# print(f"Current working directory: {os.getcwd()}")
# print(f"Contents of current directory: {os.listdir()}")
# print(f"Contents of fp_ros: {os.listdir('/home/FoundationPose_ros/install/fp_ros/lib/python3.10/site-packages/fp_ros')}")
# print(f"Contents of foundationpose: {os.listdir('/home/FoundationPose_ros/install/fp_ros/lib/python3.10/site-packages/fp_ros/foundationpose')}")

from .foundationpose.estimater import *
from .foundationpose.datareader import *
import argparse

class MultiCameraPoseEstimator(Node):
    def __init__(self):
        super().__init__('multi_camera_pose_estimator')
        
        self.initialize_parameters()
        self.initialize_buffers()
        self.create_subscribers()
        self.initialize_pose_estimation()
        
        self.create_timer(0.01, self.process_frames)
        
        self.frame_count = 0
        self.active_camera = 'main'
        self.consecutive_occlusions = 0
        self.switch_count = 0
        self.camera_priority = self.create_camera_priority_queue()
        self.payload_pose_broadcaster = TransformBroadcaster(self)
        self.frame_count = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)




    ##################################
    # Initialize parameters
    ####################################

    def initialize_parameters(self):
        self.declare_parameter('mesh_file', '')
        self.declare_parameter('debug_dir', '')
        self.declare_parameter('output_dir', '')
        self.declare_parameter('est_refine_iter', 5)
        self.declare_parameter('track_refine_iter', 2)
        self.declare_parameter('debug', 1)
        self.declare_parameter('buffer_size', 3)

        self.mesh_file = self.get_parameter('mesh_file').value
        self.debug_dir = self.get_parameter('debug_dir').value
        self.output_dir = self.get_parameter('output_dir').value
        self.est_refine_iter = self.get_parameter('est_refine_iter').value
        self.track_refine_iter = self.get_parameter('track_refine_iter').value
        self.debug = self.get_parameter('debug').value
        self.buffer_size = self.get_parameter('buffer_size').value

    def initialize_buffers(self):
        self.cv_bridge = CvBridge()
        self.color_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.depth_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.mask_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.K_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        #self.ob_in_cam_buffers = [deque(maxlen=self.buffer_size) for _ in range(3)]
        self.headers = [deque(maxlen=self.buffer_size) for _ in range(3)]

    def create_subscribers(self):
        self.subscription1 = self.create_subscription(
            RGBDSeg, 'camera1/camera1/rgbdseg', self.callback1, QoSProfile(depth=10))
        self.subscription2 = self.create_subscription(
            RGBDSeg, 'camera2/camera2/rgbdseg', self.callback2, QoSProfile(depth=10))
        self.subscription3 = self.create_subscription(
            RGBDSeg, 'camera3/camera3/rgbdseg', self.callback3, QoSProfile(depth=10))


    ##################################
    # Initialize FP data
    ####################################

    def initialize_pose_estimation(self):
        set_logging_format()
        set_seed(0)        
        mesh = trimesh.load(self.mesh_file)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        os.system(f'mkdir -p {self.debug_dir}/track_vis {self.debug_dir}/ob_in_cam')
        os.makedirs(self.output_dir, exist_ok=True)
        
        scorer = ScorePredictor()
        refiner = PoseRefinePredictor()
        glctx = dr.RasterizeCudaContext()
        
        self.est = FoundationPose(
            model_pts=mesh.vertices,
            model_normals=mesh.vertex_normals,
            mesh=mesh,
            scorer=scorer,
            refiner=refiner,
            debug_dir=self.debug_dir,
            debug=self.debug,
            glctx=glctx
        )
        logging.info("estimator initialization done")

        self.window_size = 3
        self.mtc_threshold = 0.7
        self.masks = deque(maxlen=self.window_size)
        self.N = 300        

    def callback1(self, msg):
        self.update_buffers(0, msg)

    def callback2(self, msg):
        self.update_buffers(1, msg)

    def callback3(self, msg):
        self.update_buffers(2, msg)


    ##################################
    # Retrieve and store all data async
    # to processing
    ####################################

    def update_buffers(self, camera_index, msg):
        color = self.cv_bridge.imgmsg_to_cv2(msg.rgb, desired_encoding="bgr8")
        depth = self.cv_bridge.imgmsg_to_cv2(msg.depth, desired_encoding="passthrough")
        mask = self.cv_bridge.imgmsg_to_cv2(msg.seg, desired_encoding="mono8").astype(bool)
        K = np.array(msg.camera_matrix).reshape(3, 3)
        header = msg.rgb.header
        #ob_in_cam = np.array(msg.ob_in_cam).reshape(4, 4)

        self.color_buffers[camera_index].append(color)
        self.depth_buffers[camera_index].append(depth)
        self.mask_buffers[camera_index].append(mask)
        self.K_buffers[camera_index].append(K)
        self.headers[camera_index].append(header)
        #self.ob_in_cam_buffers[camera_index].append(ob_in_cam)

    def get_latest_camera_data(self):
        return {
            'main': {
                'color': self.color_buffers[0][-1],
                'depth': self.depth_buffers[0][-1],
                'mask': self.mask_buffers[0][-1],
                'K': self.K_buffers[0][-1],
                'header': self.headers[0][-1]
                #'ob_in_cam': self.ob_in_cam_buffers[0][-1]
            },
            'second': {
                'color': self.color_buffers[1][-1],
                'depth': self.depth_buffers[1][-1],
                'mask': self.mask_buffers[1][-1],
                'K': self.K_buffers[1][-1],
                'header': self.headers[1][-1]
                #'ob_in_cam': self.ob_in_cam_buffers[1][-1]
            },
            'third': {
                'color': self.color_buffers[2][-1],
                'depth': self.depth_buffers[2][-1],
                'mask': self.mask_buffers[2][-1],
                'K': self.K_buffers[2][-1],
                'header': self.headers[2][-1]
                #'ob_in_cam': self.ob_in_cam_buffers[2][-1]
            }
        }

    ##################################
    # helper functions for mTC check and 
    # camera priority switch
    ####################################

    def has_pixels(self, mask: np.ndarray) -> bool:
        return np.sum(mask) > 30

    def calculate_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            return mask.astype(bool)        
        mask1 = ensure_binary_mask(mask1)
        mask2 = ensure_binary_mask(mask2)
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        iou = np.sum(intersection) / (np.sum(union) + 1e-6)
        return iou

    def calculate_mtc(self, masks: deque, window_size: int = 3) -> float:
        window_ious = [self.calculate_iou(masks[j], masks[j+1]) 
                    for j in range(window_size-1)]
        return np.mean(window_ious)

    def detect_occlusion(self, mtc_score: float, threshold: float = 0.7) -> bool:
        return mtc_score < threshold


    ##################################
    # Obtain the next available camera
    # with highest priority for fp after
    # checking if consistency < threshold
    ####################################

    def create_camera_priority_queue(self):
        pq = PriorityQueue()
        pq.put((1, 'main'))
        pq.put((2, 'second'))
        pq.put((3, 'third'))
        return pq

    def get_next_available_camera(self, current_camera, mask, force_switch=False):
        temp_queue = PriorityQueue()
        next_camera = None

        while not self.camera_priority.empty():
            priority, camera = self.camera_priority.get()
            if self.has_pixels(mask):
                if force_switch and camera != current_camera:
                    next_camera = camera
                    temp_queue.put((priority, camera))
                    break
                elif not force_switch and (camera != current_camera or camera == 'main'):
                    next_camera = camera
                    temp_queue.put((priority, camera))
                    break
            temp_queue.put((priority, camera))

        # Restore the queue
        while not self.camera_priority.empty():
            temp_queue.put(self.camera_priority.get())
        self.camera_priority = temp_queue

        return next_camera

    ##################################
    # Main loop for initializing processing
    ####################################

    def process_frames(self):
        if not all(len(buffer) > 0 for buffer in self.color_buffers):
            return
        
        self.frame_count += 1
        self.switch_count += 1

        camera_data = self.get_latest_camera_data()
        self.masks.append(camera_data['main']['mask'])

        if self.frame_count == 1 and self.active_camera == 'main':
            self.pose_main = self.est.register(
                K=camera_data['main']['K'],
                rgb=camera_data['main']['color'],
                depth=camera_data['main']['depth'],
                ob_mask=camera_data['main']['mask'],
                iteration=self.est_refine_iter
            )
        else:
            self.process_frame(camera_data)


        # Check for higher priority camera every N frames
        if self.switch_count % self.N == 0 and self.switch_count > 0:
            next_camera = self.get_next_available_camera(self.active_camera, camera_data[self.active_camera]['mask'], force_switch=False)
            if next_camera and next_camera != self.active_camera:
                print(f"Switching from {self.active_camera} to higher priority {next_camera} camera")
                self.switch_count = 0
                if self.active_camera == 'main':

                    self.pose_active = self.pose_transform_to_cam(camera_data['main']['header'].frame_id, 
                                                        camera_data[next_camera]['header'].frame_id, 
                                                        self.pose_main)

                    # self.pose_active = pose_transform_to_cam(camera_data['main']['cam_transform'], 
                    #                                     camera_data[next_camera]['cam_transform'], 
                    #                                     self.pose_main)
                    

                elif next_camera == 'main':

                    self.pose_main = self.pose_transform_to_cam(camera_data[self.active_camera]['header'].frame_id,
                                                        camera_data['main']['header'].frame_id, 
                                                        self.pose_active)

                    # self.pose_main = pose_transform_to_cam(camera_data[self.active_camera]['cam_transform'], 
                    #                                 camera_data['main']['cam_transform'], 
                    #                                 self.pose_active)
                else:

                    self.pose_active = self.pose_transform_to_cam(camera_data[self.active_camera]['header'].frame_id,
                                                        camera_data[next_camera]['header'].frame_id, 
                                                        self.pose_active)
                    
                    # self.pose_active = pose_transform_to_cam(camera_data[self.active_camera]['cam_transform'], 
                    #                                     camera_data[next_camera]['cam_transform'], 
                    #                                     self.pose_active)
                self.active_camera = next_camera
                self.est.pose_last = torch.from_numpy((self.pose_main if self.active_camera == 'main' else self.pose_active).reshape(1, 4, 4))
                self.masks.clear()       

        self.save_results(camera_data)


    ##################################
    # broadcast the pose
    ####################################

    def broadcast_payload_pose(self, common_header, pose):
        tf = TransformStamped()
        tf.header.stamp = common_header.stamp
        tf.header.frame_id = common_header.frame_id
        tf.child_frame_id = "payload_model"
        pose = pose.reshape(4, 4)

        translation = pose[:3, 3]
        print(f"Translation: {translation}")
        tf.transform.translation.x = float(translation[0])
        tf.transform.translation.y = float(translation[1])
        tf.transform.translation.z = float(translation[2])

        quaternion = quaternion_from_matrix(pose)
        
        tf.transform.rotation.x = quaternion[0]
        tf.transform.rotation.y = quaternion[1]
        tf.transform.rotation.z = quaternion[2]
        tf.transform.rotation.w = quaternion[3]

        self.payload_pose_broadcaster.sendTransform(tf)

    def pose_transform_to_cam(self, source_frame, target_frame, pose):
        try:
            t = self.tf_buffer.lookup_transform(target_frame, source_frame, rclpy.time.Time())
        except TransformException as e:
            print(f"Error: {e}")
            return None
        
        quat = pyqt.Quaternion(w=t.transform.rotation.w, 
                                   x=t.transform.rotation.x, 
                                   y=t.transform.rotation.y, 
                                   z=t.transform.rotation.z)
        rot = quat.rotation_matrix

        transf = np.eye(4)
        transf[:3, :3] = rot
        transf[:3, 3] = np.array([t.transform.translation.x, 
                                    t.transform.translation.y, 
                                    t.transform.translation.z])
        
        return transf @ pose
        


        



    ##################################
    # Main loop for tracking processing
    ####################################

    def process_frame(self, camera_data):
        print("ACTIVE CAMERA IS: ", self.active_camera)
        if len(self.masks) == self.window_size:
            mtc_score = self.calculate_mtc(self.masks, self.window_size)
            print(f"Current mtc_score for {self.active_camera} is {mtc_score}")
            
            if self.detect_occlusion(mtc_score, self.mtc_threshold):
                self.consecutive_occlusions += 1
                print(f"Potential occlusion detected at frame {self.frame_count} with mTC score: {mtc_score}")
                
                if self.consecutive_occlusions >= 3:
                    print("Occlusion detected in 3 consecutive frames. Switching camera.")
                    next_camera = self.get_next_available_camera(self.active_camera, camera_data[self.active_camera]['mask'], force_switch=True)
                    if next_camera and next_camera != self.active_camera:
                        # switch to alternative cam if consistency < threshold
                        print(f"Switching from {self.active_camera} to {next_camera} camera")
                        self.switch_count = 0
                        # set the previous frame to the projection of transformation matrix
                        # for fp initialization for tracking (modify previous frames' frame of ref)
                        if self.active_camera == 'main':

                            self.pose_active = self.pose_transform_to_cam(camera_data['main']['header'].frame_id, 
                                                                camera_data[next_camera]['header'].frame_id, 
                                                                self.pose_main)

                            # self.pose_active = pose_transform_to_cam(camera_data['main']['cam_transform'], 
                            #                                     camera_data[next_camera]['cam_transform'], 
                            #                                     self.pose_main)
                            

                        elif next_camera == 'main':

                            self.pose_main = self.pose_transform_to_cam(camera_data[self.active_camera]['header'].frame_id,
                                                                camera_data['main']['header'].frame_id, 
                                                                self.pose_active)

                            # self.pose_main = pose_transform_to_cam(camera_data[self.active_camera]['cam_transform'], 
                            #                                 camera_data['main']['cam_transform'], 
                            #                                 self.pose_active)
                        else:

                            self.pose_active = self.pose_transform_to_cam(camera_data[self.active_camera]['header'].frame_id,
                                                                camera_data[next_camera]['header'].frame_id, 
                                                                self.pose_active)
                            
                            # self.pose_active = pose_transform_to_cam(camera_data[self.active_camera]['cam_transform'], 
                            #                                     camera_data[next_camera]['cam_transform'], 
                            #                                     self.pose_active)
                        self.active_camera = next_camera
                        self.est.pose_last = torch.from_numpy((self.pose_main if self.active_camera == 'main' else self.pose_active).reshape(1, 4, 4))
                        self.masks.clear()
                        self.consecutive_occlusions = 0
                    else:
                        print(f"No available alternative camera found. Continuing with {self.active_camera} camera.")
            else:
                self.consecutive_occlusions = 0

        # perform the actual tracking calculation depending on active_camera
        if self.active_camera == 'main':
            self.pose_main = self.est.track_one(rgb=camera_data['main']['color'], 
                                    depth=camera_data['main']['depth'], 
                                    mask=camera_data['main']['mask'], 
                                    K=camera_data['main']['K'], 
                                    iteration=self.track_refine_iter)
        else:
            self.pose_active = self.est.track_one(rgb=camera_data[self.active_camera]['color'], 
                                        depth=camera_data[self.active_camera]['depth'], 
                                        mask=camera_data[self.active_camera]['mask'], 
                                        K=camera_data[self.active_camera]['K'], 
                                        iteration=self.track_refine_iter)

    def save_results(self, camera_data):
        # Save results and visualizef.
        [os.makedirs(f'{self.debug_dir}/ob_in_cam/{cam_name}', exist_ok=True) for cam_name in ['main', 'second', 'third']]

        if self.active_camera == 'main':
            common_header = camera_data[self.active_camera]['header']
            # common_header.frame_id = 'camera1_color_optical_frame'

            self.broadcast_payload_pose(common_header, self.pose_main)
            np.savetxt(f'{self.debug_dir}/ob_in_cam/{self.active_camera}/{self.frame_count}.txt', self.pose_main.reshape(4,4))
        else:
            # if self.active_camera == 'second':
            #     common_header = camera_data[self.active_camera]['header']
            #     # common_header.frame_id = 'camera2_color_optical_frame'
            # else:
            #     common_header = camera_data[self.active_camera]['header']
            #     common_header.frame_id = 'camera3_color_optical_frame'
            common_header = camera_data[self.active_camera]['header']
            
            self.broadcast_payload_pose(common_header, self.pose_active)
            np.savetxt(f'{self.debug_dir}/ob_in_cam/{self.active_camera}/{self.frame_count}.txt', self.pose_active.reshape(4,4))

        if self.debug >= 1:
            [os.makedirs(os.path.join(self.output_dir, cam_name), exist_ok=True) for cam_name in ['main', 'second', 'third']]
            [os.makedirs(os.path.join(self.output_dir, cam_name), exist_ok=True) for cam_name in ['main/seg', 'second/seg', 'third/seg']]

            color = camera_data[self.active_camera]['color']
            pose = self.pose_main if self.active_camera == 'main' else self.pose_active
            K = camera_data[self.active_camera]['K']

            center_pose = pose @ np.linalg.inv(self.to_origin)
            vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=self.bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=K, thickness=3, transparency=0, is_input_rgb=True)            
            cv2.imwrite(os.path.join(self.output_dir, self.active_camera, f'frame_{self.frame_count:04d}.png'), vis[...,::-1])
            cv2.imwrite(os.path.join(self.output_dir, self.active_camera, 'seg', f'frame_{self.frame_count:04d}.png'), camera_data[self.active_camera]['mask'].astype(np.uint8)*255)

            print(f"Frame {self.frame_count} saved in {self.active_camera} camera directory")

    # Helper methods (calculate_mtc, detect_occlusion, switch_camera, etc.) should be implemented here

def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraPoseEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()