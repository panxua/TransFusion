from transfusion_head import TransFusionHead

from mmdet3d.models.builder import HEADS, build_fuser, build_backbone, build_neck
import torch
from torch import nn 

@HEADS.register_module()
class BEVFusionHead(TransFusionHead):
    def __init__(self, fuser_cfg, backbone_cfg, neck_cfg, **kwargs):
        # using lidar-only arch
        if fuser_cfg:
            if 'fuse_img' in kwargs:
                kwargs['fuse_img'] = False
            self.fuser = build_fuser(fuser_cfg)
        if backbone_cfg:
            if backbone_cfg:
                self.backbone = build_backbone(backbone_cfg)

        if neck_cfg:
            if neck_cfg:
                self.neck = build_neck(neck_cfg)

        super(self, BEVFusionHead).__init__(kwargs)

    def forward(self, feats, img_feats, img_metas):
        feats = self.fuser(feats, img_feats)

        assert self.with_backbone, self.with_neck 

        if self.with_backbone:
            feats = self.backbone(feats)
        if self.with_neck:
            feats = self.neck(feats)

        super(self, BEVFusionHead).forward(feats, None, None)

    @property
    def with_fuser(self):
        """bool: Whether the fuser exists."""
        return hasattr(self, 'fuser') and self.fuser is not None
    @property
    def with_backbone(self):
        """bool: Whether the backbone exists."""
        return hasattr(self, 'backbone') and self.backbone is not None
    @property
    def with_neck(self):
        """bool: Whether the neck exists."""
        return hasattr(self, 'neck') and self.neck is not None

        

