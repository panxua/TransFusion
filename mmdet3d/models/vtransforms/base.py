from cmath import rect
from typing import Tuple

import torch
from mmcv.runner import force_fp32
from torch import nn, squeeze

from mmdet3d.ops import bev_pool

__all__ = ["BaseTransform", "BaseDepthTransform"]


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor(
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]
    )
    return dx, bx, nx


class BaseTransform(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False).cuda()
        self.bx = nn.Parameter(bx, requires_grad=False).cuda()
        self.nx = nn.Parameter(nx, requires_grad=False).cuda()

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    def init_weights(self):
        raise NotImplementedError

    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (
            torch.arange(*self.dbound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = ds.shape

        xs = (
            torch.linspace(0, iW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        ys = (
            torch.linspace(0, iH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False).cuda()

    @force_fp32()
    def get_geometry(
        self,
        rots,
        trans,
        intrins,
        post_rots,
        post_trans,
        lidar2ego_rots,
        lidar2ego_trans,
        **kwargs,
    ):
        B, N, _ = trans.shape
        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (
            torch.inverse(post_rots)
            .view(B, N, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
        )
        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        if intrins.shape[-1]==4: #for kitti
            rect_trans = intrins[..., :3, 3]
            intrins = intrins[..., :3, :3]
            assert all(rect_trans[...,-1]== 0), "LSS will has a problem"
            points -= rect_trans.view(B,N,1,1,1,3,1)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)
        # ego_to_lidar
        points -= lidar2ego_trans.view(B, 1, 1, 1, 1, 3)
        points = (
            torch.inverse(lidar2ego_rots)
            .view(B, 1, 1, 1, 1, 3, 3)
            .matmul(points.unsqueeze(-1))
            .squeeze(-1)
        )

        if "extra_rots" in kwargs:
            extra_rots = kwargs["extra_rots"]
            points = (
                extra_rots.view(B, 1, 1, 1, 1, 3, 3)
                .repeat(1, N, 1, 1, 1, 1, 1)
                .matmul(points.unsqueeze(-1))
                .squeeze(-1)
            )
        if "extra_trans" in kwargs:
            extra_trans = kwargs["extra_trans"]
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        # bx: bound[2]
        # dx: bound[0]+bound[2]/2
        # nx: bound[1]-bound[0]/bound[2]
        # bx-dx/2: bound[2] - bound[0]/2 - bound[2]/4
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.nx[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.nx[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.nx[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        img_aug_matrix,
        lidar_aug_matrix,
        metas=None,
        **kwargs,
    ):
        # TODO: need to be move to init/re-init; should not be called repeatedly
        if 'img_shape' in metas[0].keys() and metas[0]['img_shape'][:2] != self.image_size:
            self.image_size = metas[0]['img_shape'][:2]
            from math import ceil
            self.feature_size = (ceil(self.image_size[0]/8),ceil(self.image_size[1]/8))
            # if 'pad_shape' in metas[0].keys() and metas[0]['pad_shape'] != self.image_size:
            #     temp, self.image_size = self.image_size, metas[0]['pad_shape'][:2] 
            #     self.frustum = self.create_frustum()
            #     self.image_size = temp
            # else:
            self.frustum = self.create_frustum()
        
        # (ul,vl,l) = intrinsic@ego2sensor@lidar2ego@(C|1)
        # (rots @ (intrin-1 @ (ul,vl,l) + trans = (lidar_rot | lidar_trans)(c|1)  
        B,N,_,_ = lidar2image.shape
        assert N==1, "Not support yet for multi-view"
        if sensor2ego:
            rots = sensor2ego[..., :3, :3]
            trans = sensor2ego[..., :3, 3]
        else:
            rots = torch.eye(3).expand(B,N,3,3).to(img.device)
            trans = torch.zeros(3).expand(B,N,3).to(img.device)
    
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2cam_rot = lidar2camera[..., :3, :3]
        lidar2cam_trans = lidar2camera[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            rots,
            trans,
            camera_intrinsics,
            post_rots,
            post_trans,
            lidar2cam_rot,
            lidar2cam_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans
        )

        x = self.get_cam_feats(img)
        fw, fh, _ = geom.shape[-3:]
        x = self.bev_pool(geom, x[...,:fw,:fh,:])
        return x


class BaseDepthTransform(BaseTransform):
    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        # if any([meta['sample_idx']==3539 for meta in metas]):
        #     print()
        # TODO: need to be move to init/re-init; should not be called repeatedly
        if 'img_shape' in metas[0].keys() and metas[0]['img_shape'][:2] != self.image_size:
            self.image_size = metas[0]['img_shape'][:2]
            from math import ceil
            self.feature_size = (ceil(self.image_size[0]/8),ceil(self.image_size[1]/8))
            # if 'pad_shape' in metas[0].keys() and metas[0]['pad_shape'] != self.image_size:
            #     temp, self.image_size = self.image_size, metas[0]['pad_shape'][:2] 
            #     self.frustum = self.create_frustum()
            #     self.image_size = temp
            # else:
            self.frustum = self.create_frustum()
        
        # (ul,vl,l) = intrinsic@ego2sensor@lidar2ego@(C|1)
        # (rots @ (intrin-1 @ (ul,vl,l) + trans = (lidar_rot | lidar_trans)(c|1)  
        B,N,_,_ = lidar2image.shape
        assert N==1, "Not support yet for multi-view"
        if sensor2ego:
            rots = sensor2ego[..., :3, :3]
            trans = sensor2ego[..., :3, 3]
        else:
            rots = torch.eye(3).expand(B,N,3,3).to(img.device)
            trans = torch.zeros(3).expand(B,N,3).to(img.device)
    
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        lidar2cam_rot = lidar2camera[..., :3, :3]
        lidar2cam_trans = lidar2camera[..., :3, 3]
        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        batch_size, num_views, _, _ = lidar2camera.shape
        depth = torch.zeros(batch_size, num_views, 1, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            cur_coords = points[b][:, :3].transpose(1, 0)
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = (
                (cur_coords[..., 0] < self.image_size[0])
                & (cur_coords[..., 0] >= 0)
                & (cur_coords[..., 1] < self.image_size[1])
                & (cur_coords[..., 1] >= 0)
            )

            # TODO: substitude with radar depth
            for c in range(num_views):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        #TODO generalized
        geom = self.get_geometry( #modified for kitti
            rots,
            trans,
            cam_intrinsic,
            post_rots,
            post_trans,
            lidar2cam_rot,
            lidar2cam_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x
