import torch
import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path 

from matplotlib import pyplot as plt
from moviepy.editor import *
from mmdet.core import multi_apply
from ...core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode,
                         points_cam2img)


import sys
sys.path.append("/home/xuanyu/radarfusion/CenterFusion/src/tools/vod_lib")
from vod.frame import KittiLocations,FrameDataLoader,FrameLabels,FrameTransformMatrix
from vod.evaluation.evaluation_common import get_label_annotation
from vod.visualization.vis_3d import Visualization3D

DATA_DIR = "/home/xuanyu/radarfusion/CenterFusion/data/view_of_delft_PUBLIC"

class Visualizer:
    def __init__(self, root_dir=DATA_DIR, split_name = 'val', classes_visualized=['Car','Cyclist','Pedestrian'],
        work_dir = None):
        self.kitti_locations = KittiLocations(root_dir=root_dir)
        self.classes_visualized = classes_visualized
        self.max_distance_threshold = 0
        clips_info_path = Path(root_dir).joinpath("clips_info.yaml")
        clip_dir = Path(root_dir).joinpath("clips/clips")

        assert clip_dir.is_dir, "{} is not a valid directory".format(clip_dir)
        assert clips_info_path.is_file, "{} is not a valid file".format(clips_info_path)
        with open(clips_info_path,"r") as f:
            try:
                temp_dict = yaml.safe_load(f)
            except:
                raise Exception("{} is not a valid yaml file".format(clips_info_path))
        assert split_name in temp_dict, "{} is not in the file {}".format(split_name,clips_info_path)

        self.clip_names = temp_dict[split_name]
        self.clip_frame_names = {}
        for clip_name in self.clip_names:
            clip_info_path = clip_dir.joinpath("{}.txt".format(clip_name))
            assert clip_info_path.is_file, "{} is not in {}".format(clip_name,clip_dir)
            with open(clip_info_path,"r") as f:
                clip_info = f.readlines()
            self.clip_frame_names[clip_name] = list(map(str.strip, clip_info))
    
    def show_results_frames(self, pred_bboxes, gt_bboxes = None):
        multi_apply(self.show_results_frames_single, pred_bboxes, gt_bboxes)

    def show_results_frames_single(self, pred_bbox, gt_bbox=None, ):
        sample_idx = pred_bbox['sample_idx']
        frame_name = '%05d'%sample_idx[0]
        frame_data = FrameDataLoader(kitti_locations=self.kitti_locations,frame_number=frame_name)
        frame_img = frame_data.image
        frame_trans = FrameTransformMatrix(frame_data)

        if pred_bbox:
            torch.cat([gt_bbox['location'],gt_bbox['dimentsions'],gt_bbox['rotation_y']],dim=1)
            # np.concatenate([loc, dims, rots[..., np.newaxis]],
            #                           axis=1).astype(np.float32
            gt_bbox = CameraInstance3DBoxes()
            gt_bbox = self._get_img_corners(gt_bbox['location'], gt_bbox['dimensions'], frame_trans)


    def _get_3d_corners(self, center, dim, alpha, frame_trans):
        x = [dim[2]/2, dim[2]/2, -dim[2]/2, -dim[2]/2, dim[2]/2, dim[2]/2, -dim[2]/2, -dim[2]/2,]
        y = [dim[1]/2, -dim[1]/2, -dim[1]/2, dim[1]/2, dim[1]/2, -dim[1]/2, -dim[1]/2, dim[1]/2,]
        z = [0, 0, 0, 0, dim[0], dim[0], dim[0], dim[0]]
    
    # def _get_img_corners(self, center, dim, frame_trans):
    #         bboxes = []
    #         corners_3d = get_3d_label_corners(labels)

    #         for index, label in enumerate(labels.labels_dict):
    #             rotation = -(label['rotation'] + np.pi / 2)  # undo changes made to rotation
    #             rot_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
    #                                 [np.sin(rotation), np.cos(rotation), 0],
    #                                 [0, 0, 1]])

    #             center = (transformations_matrix.t_lidar_camera @ np.array([label['x'],
    #                                                                         label['y'],
    #                                                                         label['z'],
    #                                                                         1]))[:3]

    #             new_corner_3d = np.dot(rot_matrix, corners_3d[index]['corners_3d']).T + center 
    #             new_corners_3d_hom = np.concatenate((new_corner_3d, np.ones((8, 1))), axis=1)
    #             new_corners_3d_hom = transformations.homogeneous_transformation(new_corners_3d_hom,
    #                                                                                 transformations_matrix.t_camera_lidar)

    #             corners_img = np.dot(new_corners_3d_hom, transformations_matrix.camera_projection_matrix.T)
    #             corners_img = (corners_img[:, :2].T / corners_img[:, 2]).T
    #             corners_img = corners_img.tolist()
    #             # if crop:
    #             #     for corners in corners_img:


    #             distance = np.linalg.norm((label['x'], label['y'], label['z']))

    #             bboxes.append({'label_class': label['label_class'],
    #                         'corners': corners_img,
    #                         'score': label['score'],
    #                         'range': distance})

    #         # if isSorted:
    #         #     bboxes = sorted(bboxes, key=lambda d: d['range'])

    #         return bboxes

def show_result_offline(gt_bboxes, pred_bboxes, show=True): 
    '''
    for kitti:
    dict_keys(['name', 'truncated', 'occluded', 'alpha', 'bbox', 'dimensions', 'location', 'rotation_y', 'score', 'index', 'group_ids', 'difficulty', 'num_points_in_gt']
    '''
    Visualizer().show_results_frames(pred_bboxes, gt_bboxes)

