import mmcv
import torch
from mmcv.parallel import DataContainer as DC
from mmcv.runner import force_fp32
from os import path as osp
from torch import nn as nn
from torch.nn import functional as F

from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.ops import Voxelization
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from .. import builder
from .mvx_two_stage import MVXTwoStageDetector
from .transfusion import TransFusionDetector


@DETECTORS.register_module()
class RCFusion(TransFusionDetector):
    def __init__(self, 
                 freeze_img_backbone=False,
                 freeze_img_neck=False,
                 img_vtransform=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 **kwargs):

        self.freeze_img_backbone = freeze_img_backbone
        self.freeze_img_neck = freeze_img_neck

        super(RCFusion, self).__init__(**kwargs)

        if img_vtransform:
            self.img_vtransform = builder.build_vtransform(
                img_vtransform)
        if img_bev_encoder_backbone:
            self.img_bev_encoder_backbone = builder.build_backbone(
                img_bev_encoder_backbone)
        if img_bev_encoder_neck:
            self.img_bev_encoder_neck = builder.build_neck(
                img_bev_encoder_neck)

        self.init_weights(pretrained=kwargs.get('pretrained', None))

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        super(RCFusion, self).init_weights(pretrained)
        if self.with_img_backbone:
            if not pretrained==None: 
                self.img_backbone.init_weights(pretrained=pretrained['img_backbone'])

        if self.with_img_vtransform:
            self.img_vtransform.init_weights()
            
        if self.freeze_img_backbone:
            if self.with_img_backbone:
                for param in self.img_backbone.parameters():
                    param.requires_grad = False

        if self.freeze_img_neck:
            if self.with_img_neck:
                for param in self.img_neck.parameters():
                    param.requires_grad = False

    def get_sparse_img(self, pts, img_feats, img_metas):
        
        img_meta = img_metas[0]
        lidar2img = img_meta['lidar2img'][0]
        lidar2img = torch.Tensor(lidar2img).cuda()
        img_aug_matrix = torch.eye(4).cuda()
        if 'scale_factor' in img_meta:
            img_aug_matrix[0,0] *= img_meta['scale_factor'][0]
            img_aug_matrix[1,1] *= img_meta['scale_factor'][1]
        image_size = img_metas[0]['input_shape']
        B, C, H, W = img_feats[0].shape
        feat_down_factor = image_size[0]/H
        # depth, RCS, doppler
        sparse_img = torch.zeros((B, 3, H, W)).cuda()
        for b in range(B):
            cur_coords, on_img, depth = self.project_pts_to_image(pts[b], lidar2img, img_aug_matrix, image_size)
            masked_coords = (cur_coords[:,on_img]/feat_down_factor).long()
            sparse_img[b, 0, masked_coords[0], masked_coords[1]] = depth[on_img]
            sparse_img[b, 1:3, masked_coords[0], masked_coords[1]] = pts[b][on_img, 3:5].transpose(1,0)
        return sparse_img

    def extract_img_feat(self, img, img_metas, points):
        """Extract features of images."""
        fov_feats, bev_feats = None, None
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            B, N, C, H, W = img.size()
            if img.dim() == 5 and B == 1:
                img.squeeze_(0)
            elif img.dim() == 5 and B > 1:
                img = img.view(B * N, C, H, W)
            img_feats = self.img_backbone(img.float())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        fov_feats = img_feats

        # project points to perspective sparse image
        sparse_img = self.get_sparse_img(points, img_feats, img_metas)
        img_feats = torch.cat((img_feats[0], sparse_img), dim=1)
        # project perspective features to BEV
        if self.with_img_vtransform:
            if not isinstance(img_feats,torch.Tensor):
                img_feats = img_feats[0]
            _, C, H, W = img_feats.size()
            img_feats = img_feats.view(B, N, C, H, W)
            img_feats = self.transform_img_feat(points, img_feats, img_metas)

        if self.with_img_bev_encoder_backbone:
            img_feats = self.img_bev_encoder_backbone(img_feats)
        if self.with_img_bev_encoder_neck:
            img_feats = self.img_bev_encoder_neck(img_feats)
        bev_feats = img_feats
        return fov_feats[0], [None, [bev_feats]], img_metas

    def transform_img_feat(self, pts, img_feats, img_metas):
        assert 'flip' not in img_metas[0].keys() or not img_metas[0]['flip'], "not implemented"
        assert 'pcd_horizontal_flip' not in img_metas[0].keys() or not img_metas[0]['pcd_horizontal_flip'], "not implemented"
        assert  'pcd_horizontal_flip' not in img_metas[0].keys() or not img_metas[0]['pcd_vertical_flip'], "not implement"
        
        lidar2camera, lidar2image, intrins, img_aug_matrices = [],[],[],torch.Tensor(0,4,4)

        # TODO time consuming
        # prepare image meta
        for img_meta in img_metas:
            lidar2camera += img_meta['lidar2cam']
            lidar2image += img_meta['lidar2img']
            intrins += img_meta['intrins']
            img_aug_matrix = torch.eye(4)
            if 'scale_factor' in img_meta:
                # TODO the scale factor is not correct, incorporating tiny mismatches
                img_aug_matrix[0,0] *= img_meta['scale_factor'][0]
                img_aug_matrix[1,1] *= img_meta['scale_factor'][1]
            img_aug_matrices = torch.cat([img_aug_matrices,img_aug_matrix.unsqueeze(0)],0)

        lidar2camera = torch.Tensor(lidar2camera).cuda().unsqueeze(1)
        lidar2image = torch.Tensor(lidar2image).cuda().unsqueeze(1)
        intrins = torch.Tensor(intrins).cuda().unsqueeze(1)
        img_aug_matrices = img_aug_matrices.cuda().unsqueeze(1)

        img_feats = self.img_vtransform(
            img_feats,
            pts,
            None,
            None,
            lidar2camera,
            lidar2image,
            intrins,
            img_aug_matrices,
            torch.eye(4).expand(img_aug_matrices.shape).to(img_aug_matrices.device),
            img_metas
        )
        return img_feats

    def extract_pts_feat(self, pts):
        """Extract features of points."""
        if not self.with_pts_backbone:
            return None
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,
                                                )
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x

    def project_pts_to_image(self, pts, lidar2img, img_aug_matrix, image_size):

        res_coords = pts[:, :3].transpose(1, 0)
        # projects points from radar onto images 
        cur_coords = lidar2img[:3, :3].matmul(res_coords)
        cur_coords += lidar2img[:3, 3].unsqueeze(1)
        # get 2d coords
        dist = cur_coords[2, :]
        cur_coords[2, :] = torch.clamp(dist, 1e-5, 1e5)
        cur_coords[:2, :] /= cur_coords[2, :]
        # imgaug
        cur_coords = img_aug_matrix[:3, :3].matmul(cur_coords)
        cur_coords += img_aug_matrix[:3, 3].unsqueeze(1)
        depth = cur_coords[2, :]
        # normalize coords for grid sample
        cur_coords = cur_coords[[1, 0]]
        on_img = (
            (cur_coords[0] < image_size[0])
            & (cur_coords[0] >= 0)
            & (cur_coords[1] < image_size[1])
            & (cur_coords[1] >= 0)
        )
        return cur_coords, on_img, depth 

    def fetch_img_features(self, pts, img_feats, img_metas):
        
        aug_pts = []
        img_meta = img_metas[0]
        lidar2img = img_meta['lidar2img'][0]
        lidar2img = torch.Tensor(lidar2img).cuda()
        img_aug_matrix = torch.eye(4).cuda()
        if 'scale_factor' in img_meta:
            img_aug_matrix[0,0] *= img_meta['scale_factor'][0]
            img_aug_matrix[1,1] *= img_meta['scale_factor'][1]
        image_size = img_metas[0]['input_shape']
        B, C, H, W = img_feats.shape
        feat_down_factor = image_size[0]/H

        for b in range(B):
            N = pts[b].shape[0]
            cur_coords, on_img, _ = self.project_pts_to_image(pts[b], lidar2img, img_aug_matrix, image_size)
            masked_coords = (cur_coords[:,on_img]/feat_down_factor).long()
            fetch_img_feat = torch.zeros(C, N).cuda()
            fetch_img_feat[:, on_img] = img_feats[b, :, masked_coords[0],masked_coords[1]]
            aug_pts.append(torch.cat((pts[b],fetch_img_feat.transpose(1,0)), dim=1))
        
        return aug_pts

    def extract_feat(self, points, img, img_metas):
        """Extract features from images and points"""
        fov_feats, img_feats, img_metas = self.extract_img_feat(img, img_metas, points)
        points_aug = self.fetch_img_features(points, fov_feats, img_metas)
        pts_feats = self.extract_pts_feat(points_aug)
        
        return (img_feats, pts_feats)

    def forward_pts_train(self,
                          pts_feats,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(pts_feats, img_feats, img_metas)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses
        
    def forward_img_train(self,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        outs = self.pts_bbox_head(img_feats)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs, img_metas]
        losses = self.pts_bbox_head.loss(*loss_inputs)
        return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        losses = dict()
        if pts_feats:
            losses_pts = self.forward_pts_train(pts_feats, img_feats, gt_bboxes_3d,
                                                gt_labels_3d, img_metas,
                                                gt_bboxes_ignore)
            losses.update(losses_pts)
        elif img_feats:
            if self.with_pts_bbox:
                losses_img = self.forward_img_train(img_feats, gt_bboxes_3d,
                                                    gt_labels_3d, img_metas,
                                                    gt_bboxes_ignore)
            else:
                losses_img = super().forward_img_train(
                    img_feats,
                    img_metas=img_metas,
                    gt_bboxes=gt_bboxes,
                    gt_labels=gt_labels,
                    gt_bboxes_ignore=gt_bboxes_ignore,
                    proposals=proposals)
            losses.update(losses_img)

        return losses

    def simple_test_img(self, x, img_metas, rescale=False,**kwargs):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x) # reg, height ...
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale) # decode to bboxes
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]

        # for debug #
        # print('warning debuging')
        # heatmaps, anno_boxes, inds, masks = self.pts_bbox_head.get_targets(
        #     kwargs['gt_bboxes_3d'][0], kwargs['gt_labels_3d'][0])
        # targets = []
        # anno_boxes = anno_boxes
        # for task_id, preds_dict in enumerate(anno_boxes):
        #     # heatmap focal loss
        #     regs = torch.zeros(1,2,64,64).cuda()
        #     heights = torch.zeros(1,1,64,64).cuda()
        #     dim = torch.zeros(1,3,64,64).cuda()
        #     rot = torch.zeros(1,2,64,64).cuda()
        #     for i, ind in enumerate(inds[task_id][0][masks[task_id][0]!=0]):
        #         y = ind//64
        #         x = ind%64
        #         #TODO check
        #         regs[0,:,y,x] = preds_dict[0][i][:2]
        #         heights[0,:,y,x] = preds_dict[0][i][2]
        #         dim[0,:,y,x] = preds_dict[0][i][3:6]
        #         rot[0,:,y,x]= preds_dict[0][i][6:8]
        #     targets.append([{'reg':regs,'height':heights, 'dim':dim, 'rot':rot, 'heatmap':heatmaps[task_id]}])
        # temp_bbox_list = self.pts_bbox_head.get_bboxes(
        #     targets, img_metas, rescale=rescale) # decode to bboxes
        # temp_bbox_results = [
        #     bbox3d2result(bboxes, scores, labels)
        #     for bboxes, scores, labels in temp_bbox_list
        # ]

        return bbox_results

    def simple_test(self, points, img_metas, img=None, rescale=False, **kwargs):
        """Test function without augmentaiton."""
        img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(
                pts_feats, img_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        elif img_feats and self.with_pts_bbox:
            bbox_img = self.simple_test_img(
                img_feats, img_metas, rescale=rescale, **kwargs)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['pts_bbox'] = img_bbox
        elif img_feats:
            bbox_img = super().simple_test_img(
                img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        return bbox_list

    @property
    def with_img_vtransform(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_vtransform') and self.img_vtransform is not None
    @property
    def with_img_bev_encoder_backbone(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_bev_encoder_backbone') and self.img_bev_encoder_backbone is not None
    @property
    def with_img_bev_encoder_neck(self):
        """bool: Whether the detector has a 2D image backbone."""
        return hasattr(self, 'img_bev_encoder_neck') and self.img_bev_encoder_neck is not None
