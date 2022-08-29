import copy
from sys import implementation
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init, xavier_init
from mmcv.runner import force_fp32
from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.utils import FFN, PositionEmbeddingLearned, TransformerDecoderLayer
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_batch


@HEADS.register_module()
class RCFusionHead(nn.Module):
    def __init__(self,
                 fuse_img=False,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=128,
                 auxiliary=True,
                 in_channels=128 * 3,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_decoder_layers=3,
                 num_fusion_layers=1,
                 num_heads=8,
                 learnable_query_pos=False,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_consistency=dict(type='L1Loss', reduction='mean'),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 ):

        super(RCFusionHead, self).__init__()
        
        # new hyperparameter
        self.pos_thres = 0.1
        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.num_fusion_layers = num_fusion_layers
        self.bn_momentum = bn_momentum
        self.learnable_query_pos = learnable_query_pos
        self.initialize_by_heatmap = initialize_by_heatmap
        self.nms_kernel_size = nms_kernel_size
        if self.initialize_by_heatmap is True:
            assert self.learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_heatmap = build_loss(loss_heatmap)
        self.loss_consistency = build_loss(loss_consistency)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        self.fuse_img = fuse_img

        # a shared convolution for point cloud features
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            int(hidden_channel),
            kernel_size=3,
            padding=1,
            bias=bias,
        )
        # heatmap for single modality
        sm_layers = []
        sm_layers.append(ConvModule(
            hidden_channel,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        ))
        sm_layers.append(build_conv_layer(
            dict(type='Conv2d'),
            hidden_channel,
            num_classes,
            kernel_size=3,
            padding=1,
            bias=bias,
        ))
        self.heatmap_head_radar = nn.Sequential(*sm_layers)
        if self.fuse_img:
            self.heatmap_head_img = copy.deepcopy(self.heatmap_head_radar)
        
        if self.fuse_img:
            self.out_size_factor_img = out_size_factor_img
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,  # channel of img feature map
                int(hidden_channel/2),
                kernel_size=3,
                padding=1,
                bias=bias)
            
        # initilize layers after fusion
        if fuse_img:
            hidden_channel *= 2

        layers = []
        layers.append(ConvModule(
            hidden_channel,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
            conv_cfg=dict(type='Conv2d'),
            norm_cfg=dict(type='BN2d'),
        ))
        layers.append(build_conv_layer(
            dict(type='Conv2d'),
            hidden_channel,
            num_classes,
            kernel_size=3,
            padding=1,
            bias=bias,
        ))
        self.heatmap_head = nn.Sequential(*layers)

        # fuser layer
        self.fuse_layers = build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,  # channel of img feature map
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias)

        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
        
        # transformer decoder layers/prediction heads for object query with fused feature
        self.decoder = nn.ModuleList()
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            # transformer decoder layer
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                ))
            #Prediction Heads
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))

        if self.fuse_img:
            self.out_size_factor_img = out_size_factor_img
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,  # channel of img feature map
                int(hidden_channel/2),
                kernel_size=3,
                padding=1,
                bias=bias)
                
        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
        self.lidar_bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = {'fov':None, 'bev':None}
        self.img_feat_collapsed_pos = {'fov':None, 'bev':None}
        self.img_feat_shrink_pos = None
        self.img_bev_downsample_pos = None

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        for m in self.shared_conv.modules():
            if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
        if self.fuse_img:
            for m in self.shared_conv_img.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    
    def decode_heatmap(self, heatmap, bev_feat_flatten, bev_pos):

        padding = self.nms_kernel_size // 2
        local_max = torch.zeros_like(heatmap)
        # equals to nms radius = voxel_size * out_size_factor * kenel_size
        local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
        ## for Pedestrian & Traffic_cone in nuScenes
        if self.test_cfg['dataset'] == 'nuScenes':
            local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
            local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)
        elif self.test_cfg['dataset'] in ['Waymo']:  # for Pedestrian & Cyclist in Waymo 
            local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
            local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
        heatmap = heatmap * (heatmap == local_max)
        heatmap = heatmap.view(self.batch_size, heatmap.shape[1], -1)

        # top #num_proposals among all classes
        top_proposals = heatmap.contiguous().view(self.batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
        top_proposals_class = top_proposals // heatmap.shape[-1]
        top_proposals_index = top_proposals % heatmap.shape[-1]
        query_feat = bev_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, bev_feat_flatten.shape[1], -1), dim=-1)
        self.query_labels = top_proposals_class

        # add category embedding
        one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
        query_cat_encoding = self.class_encoding(one_hot.float())
        query_feat += query_cat_encoding
        query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)

        return query_feat, query_pos, top_proposals_index

    def forward_single(self, inputs, fov_inputs, bev_inputs, img_metas):
        """Forward function for CenterPoint.

        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)

        Returns:
            list[dict]: Output results for tasks.
        """
        self.batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)
        # if any([meta['sample_idx']==3539 for meta in img_metas]):
        #     print()
        #################################
        # Fuse image and lidar features to BEV features
        #################################
        bev_pos = self.lidar_bev_pos.repeat(self.batch_size, 1, 1).to(lidar_feat.device)

        if self.fuse_img:
            img_inputs = bev_inputs
            img_feat = self.shared_conv_img(bev_inputs)  # [BS * n_views, C, H, W]
            bev_feat = self.fuse_layers(torch.cat((lidar_feat, img_feat), dim=1))
        else:
            bev_feat = lidar_feat 
        bev_feat_flatten = bev_feat.view(self.batch_size, bev_feat.shape[1], -1)  # [BS, C, H*W]

        #################################
        # generate heatmap per modality
        #################################
        if self.fuse_img:
            radar_heatmap = self.heatmap_head_radar(lidar_feat)
            img_heatmap = self.heatmap_head_img(img_feat)

        #################################
        # initialize with heatmap
        #################################
        if self.initialize_by_heatmap:
            fuse_heatmap = self.heatmap_head(bev_feat)
            heatmap = fuse_heatmap.detach().sigmoid()
            
            #################
            # decode heatmap
            #################
            query_feat, query_pos, top_proposals_index = self.decode_heatmap(heatmap, bev_feat_flatten, bev_pos)


        #################################
        # transformer decoder layer (BEV feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](query_feat, bev_feat_flatten, query_pos, bev_pos)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            first_res_layer = res_layer
            ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)


        if self.initialize_by_heatmap:
            heatmap = heatmap.view(self.batch_size, heatmap.shape[1], -1)
            ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
            ret_dicts[0]['dense_heatmap'] = fuse_heatmap
        
        if self.fuse_img:
            ret_dicts[0]['radar_heatmap'] = radar_heatmap
            ret_dicts[0]['image_heatmap'] = img_heatmap
        
        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            if key not in ['dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score', 'view_type']:
                new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
            else:
                new_res[key] = ret_dicts[0][key]
        return [new_res]

    def forward(self, feats, img_feats, img_metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if img_feats is None:
            img_feats = [[None],[None]]
        for i, img_feat in enumerate(img_feats):
            if img_feat==None:
                img_feats[i]=[None]
        res = multi_apply(self.forward_single, feats, img_feats[0], img_feats[1], [img_metas])
        assert len(res) == 1, "only support one level features."
        return res

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                if key is "dense_heatmap":
                    pred_dict[key] = [preds_dict[0][key][i][batch_idx:batch_idx + 1] for i in range(len(preds_dict[0][key]))]
                else:   
                    pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, np.arange(len(gt_labels_3d)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap
        else:
            return labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)

        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers 
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1)]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d, score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, None, gt_labels_3d, self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat([gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]], dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1], feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']

                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                center_int = center.to(torch.int32)
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)

        ## response consistency from two modalities
        # gt_heatmap = copy.deepcopy(heatmap)
        # rd_heatmap = copy.deepcopy(preds_dict['radar_heatmap'][0].detach().sigmoid())
        # img_heatmap = copy.deepcopy(preds_dict['image_heatmap'][0].detach().sigmoid())
        # gt_conf = gt_heatmap.max(dim=0)
        # rd_conf = rd_heatmap.max(dim=0)
        # img_conf = img_heatmap.max(dim=0)
        # gt_p = gt_conf>self.pos_thres
        
        # from mmdet3d.core import LiDARInstance3DBoxes
        # from src.vis_utils import *
        
        # corners = LiDARInstance3DBoxes(bboxes_tensor).corners
        # corners -= torch.tensor([0,-25.6, 0]).cuda().reshape(1,1,3).repeat(600,8,1)
        # corners /= 0.1 * 8
        # plt.gca().set_xlim([0,64])
        # plt.gca().set_ylim([0,64])
        # show_heatmap(heatmap[0,:,:].cpu().detach(),"output/heatmaps/pred_Car", bboxes = corners.cpu().detach()[-300:,:,:])
        # show_heatmap(heatmap[0,:,:].cpu().detach(),"output/heatmaps/pred_Car_top20", bboxes = corners.cpu().detach()[300+torch.topk(score[0,0,-300:],20).indices,:,:])

        # corners = LiDARInstance3DBoxes(gt_bboxes_tensor).corners
        # corners -= torch.tensor([0,-25.6, 0]).cuda().reshape(1,1,3).repeat(len(gt_bboxes_tensor),8,1)
        # corners /= 0.1 * 8
        # plt.gca().set_xlim([0,64])
        # plt.gca().set_ylim([0,64])
        # show_heatmap(heatmap[0,:,:].cpu().detach(),"output/heatmaps/gt_Car", bboxes = corners.cpu().detach()[:,:,:])

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]

        
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts, **kwargs):
        """Loss function for CenterHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (list[list[dict]]): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        
        labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap = \
                         self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts[0])
        preds_dict = preds_dicts[0][0]
        loss_dict = dict()

        # compute response consistency loss
        if self.fuse_img:
        
            image_heatmap = clip_sigmoid(preds_dict['image_heatmap'])
            radar_heatmap = clip_sigmoid(preds_dict['radar_heatmap'])
            radar_tp = torch.logical_and(heatmap>self.pos_thres, radar_heatmap.detach()>self.pos_thres)
            con_weights = heatmap.new_zeros(heatmap.size())
            con_weights[radar_tp] = 1.0
            loss_consistency = self.loss_consistency(image_heatmap, radar_heatmap, con_weights, avg_factor=max(con_weights.eq(1).float().sum().item(), 1))
            loss_consistency += self.loss_consistency(radar_heatmap, heatmap, con_weights, avg_factor=max(con_weights.eq(1).float().sum().item(), 1))
            loss_consistency += self.loss_consistency(image_heatmap, heatmap, con_weights, avg_factor=max(con_weights.eq(1).float().sum().item(), 1))
            loss_dict['loss_consistency'] = loss_consistency
        
        # compute heatmap loss
        loss_heatmap = 0
        loss_heatmap = self.loss_heatmap(clip_sigmoid(preds_dict['dense_heatmap']), heatmap, avg_factor=max(heatmap.eq(1).float().sum().item(), 1))
        loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        if self.auxiliary:
            num_layers = self.num_decoder_layers 
        else:
            num_layers = 1

        for idx_layer in range(num_layers):
            if (idx_layer == self.num_decoder_layers - 1) or (idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_label_weights = label_weights[..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals].reshape(-1)
            layer_score = preds_dict['heatmap'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)
            layer_loss_cls = self.loss_cls(layer_cls_score, layer_labels, layer_label_weights, avg_factor=max(num_pos, 1))

            layer_center = preds_dict['center'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_height = preds_dict['height'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_rot = preds_dict['rot'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            layer_dim = preds_dict['dim'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][..., idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals]
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
            layer_bbox_targets = bbox_targets[:, idx_layer * self.num_proposals:(idx_layer + 1) * self.num_proposals, :]
            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

            # layer_iou = preds_dict['iou'][..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals].squeeze(1)
            # layer_iou_target = ious[..., idx_layer*self.num_proposals:(idx_layer+1)*self.num_proposals]
            # layer_loss_iou = self.loss_iou(layer_iou, layer_iou_target, layer_bbox_weights.max(-1).values, avg_factor=max(num_pos, 1))

            # from mmdet3d.core import LiDARInstance3DBoxes
            # from src.vis_utils import *

            # code = preds[0,:,:-1]
            # code[:,-1] = torch.atan2(preds[0,:,-2],preds[0,:,-1])
            # show_heatmap(heatmap.cpu().detach()[0,0,:,:], "output/heatmaps/pred_Car",bboxes=LiDARInstance3DBoxes(code).corners.cpu().detach())
            # show_heatmap(heatmap.cpu().detach()[0,0,:,:], "output/heatmaps/pred_Car_top20",bboxes=LiDARInstance3DBoxes(code[torch.topk(layer_cls_score[:300,0],20).indices,:]).corners.cpu().detach())
            # code = layer_bbox_targets[0,:,:-1]
            # code[:,-1] = torch.atan2(layer_bbox_targets[0,:,-2],layer_bbox_targets[0,:,-1])
            # show_heatmap(heatmap.cpu().detach()[0,0,:,:], "output/heatmaps/gt_Car",bboxes=LiDARInstance3DBoxes(code).corners.cpu().detach())
        


            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            

        loss_dict[f'matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.

        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer & each batch
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid())
            one_hot = F.one_hot(self.query_labels, num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0]['query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]
            temp = self.bbox_coder.decode(batch_score, batch_rot, batch_dim, batch_center, batch_height, batch_vel, filter=True)
            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(num_class=8, class_names=[], indices=[0, 1, 2, 3, 4, 5, 6, 7], radius=-1),
                    dict(num_class=1, class_names=['pedestrian'], indices=[8], radius=0.175),
                    dict(num_class=1, class_names=['traffic_cone'], indices=[9], radius=0.175),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(num_class=1, class_names=['Car'], indices=[0], radius=0.7),
                    dict(num_class=1, class_names=['Pedestrian'], indices=[1], radius=0.7),
                    dict(num_class=1, class_names=['Cyclist'], indices=[2], radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                ## adopt circle nms for different categories
                if self.test_cfg['nms_type'] != None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat([boxes3d[task_mask][:, :2], scores[:, None][task_mask]], dim=1)
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    )
                                )
                            else:
                                boxes_for_nms = xywhr2xyxyr(img_metas[i]['box_type_3d'](boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_gpu(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(bboxes=boxes3d[keep_mask], scores=scores[keep_mask], labels=labels[keep_mask])
                else:  # no nms
                    if 'post_maxsize' in self.test_cfg and self.test_cfg['post_maxsize']:
                        if self.test_cfg['post_maxsize'] < len(boxes3d):
                            topk_ind = torch.topk(scores, self.test_cfg['post_maxsize']).indices
                        else:
                            topk_ind = range(len(boxes3d))
                    else:
                        topk_ind = range(len(boxes3d))

                    ret = dict(bboxes=boxes3d[topk_ind], scores=scores[topk_ind], labels=labels[topk_ind])

                ret_layer.append(ret)
            rets.append(ret_layer)
        assert len(rets) == 1
        assert len(rets[0]) == 1
        res = [[
            img_metas[0]['box_type_3d'](rets[0][0]['bboxes'], box_dim=rets[0][0]['bboxes'].shape[-1]),
            rets[0][0]['scores'],
            rets[0][0]['labels'].int()
        ]]
        return res
