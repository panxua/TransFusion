point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
class_names = ['Car', 'Pedestrian', 'Cyclist']
# TODO change to 0.125
# voxel_size = [0.1, 0.1, 0.15]
voxel_size = [0.1, 0.1, 0.15]
out_size_factor = 8
evaluation = dict(interval=1)
dataset_type = 'VODDataset'
data_root = 'data/view_of_delft_PUBLIC/radar'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False)
img_scale = (448, 800) #TODO wrong
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
numC_Trans = 256

train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3, 5],
        # use_dim=[0, 1, 2, 3, 5],
    ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=10,
    #     use_dim=[0, 1, 2, 3, 4],
    # ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='LoadMultiViewImageFromFiles'),
    # dict(type='ObjectSample',
    #     db_sampler=dict(
    #         data_root=data_root,
    #         info_path=data_root + '/vod_radar_dbinfos_train.pkl',
    #         rate=1.0,
    #         prepare=dict(
    #             filter_by_difficulty=[-1],
    #             filter_by_min_points=dict(Car=5, Pedestrian=5, Cyclist=5)),
    #         classes=class_names,
    #         sample_groups=dict(Car=15, Pedestrian=10, Cyclist=10),
    #         points_loader=dict(
    #             type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=[0, 1, 2, 3, 4]))
    #     ),
    # dict(
    #     type='GlobalRotScaleTrans',
    #     rot_range=[-0.3925 * 2, 0.3925 * 2],
    #     scale_ratio_range=[0.9, 1.1],
    #     translation_std=[0.5, 0.5, 0.5]),
    # dict(
    #     type='RandomFlip3D',
    #     sync_2d=True,
    #     flip_ratio_bev_horizontal=0.5,
    #     flip_ratio_bev_vertical=0.5),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MyNormalize', **img_norm_cfg),
    dict(type='MyPad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3, 5],
        # use_dim=[0, 1, 2, 3, 5],
    ),
    # dict(
    #     type='LoadPointsFromMultiSweeps',
    #     sweeps_num=1,
    #     use_dim=[0, 1, 2, 3, 4],
    # ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=img_scale,
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
            dict(type='MyNormalize', **img_norm_cfg),
            dict(type='MyPad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'])
        ])
]
data = dict(
    # samples_per_gpu=1,
    # workers_per_gpu=1,

    samples_per_gpu=3,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/vod_radar_infos_train.pkl',
            split = "training",
            # load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/vod_radar_infos_val.pkl',
        split = "training",
        # load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/vod_radar_infos_val.pkl',
        split = "training",
        # load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
    # test=dict(
    #     type=dataset_type,
    #     data_root=data_root,
    #     ann_file=data_root + '/vod_radar_infos_test.pkl',
    #     split = "testing",
    #     # load_interval=1,
    #     pipeline=test_pipeline,
    #     classes=class_names,
    #     modality=input_modality,
    #     test_mode=True,
    #     box_type_3d='LiDAR'))
model = dict(
    type='BEVTransFusionDetector',
    freeze_img=False,
    freeze_bev_encoder=False,
    # img_backbone=dict(
    #     type='DLASeg',
    #     num_layers=34,
    #     heads={},
    #     head_convs=-1,
    #     ),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),    
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,   
        num_outs=3,
        norm_cfg=dict(type="BN2d",requires_grad=True),
        act_cfg=dict(type="ReLU",inplace=True),
        upsample_cfg=dict(mode="bilinear",align_corners=False)),
    img_vtransform=dict(
        type="DepthLSSTransform",
        in_channels=256,
        out_channels=256, #original 80
        image_size=img_scale,
        feature_size= (img_scale[0]//8,img_scale[1]//8),
        xbound=[0, 51.2, 0.4],
        ybound=[-25.6, 25.6, 0.4],
        zbound=[-3.0, 2.0, 5.0], #[-10,10,20]
        dbound=[1.0, 60.0, 0.5],
        downsample=2),
    # img_bev_encoder_backbone=dict(
    #     type='ResNetForBEVDet',
    #     numC_input=numC_Trans #256
    # ),
    # img_bev_encoder_neck=dict(
    #     type='FPN_LSS',
    #     in_channels=numC_Trans*8+numC_Trans*2, #2560
    #     out_channels=256
    # ),
    pts_voxel_layer=dict(
        max_num_points=5, #10
        voxel_size=voxel_size,
        max_voxels= 150000, #(120000, 160000),
        point_cloud_range=point_cloud_range),
    # pts_voxel_encoder=dict(
    #     type='HardSimpleVFE',
    #     num_features=5,
    #     # feat_channels=[64],
    # ),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        with_cluster_center=False,
        with_voxel_center=False,
        voxel_size=voxel_size,
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        point_cloud_range=point_cloud_range,
    ),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        # sparse_shape = [41, 1504, 1504], #modified
        sparse_shape=[41, 512, 512],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128, 128)),
        encoder_paddings=((0, 0, 1), (0, 0, 1), (0, 0, [0, 1, 1]), (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='TransFusionHead',
        # img fusion
        fuse_img=True,
        img_guidance = False,
        order=['fov', 'bev'],
        fuse_fov=False,
        fuse_bev=True,
        fuse_bev_collapse=False,
        # fuse_img_decoder=False,
        num_views=1,
        in_channels_img=256, # modified 256
        # same as lidar only
        num_proposals=300,
        auxiliary=True,
        in_channels=256 * 2,
        hidden_channel=128,
        out_size_factor_img=4,
        num_classes=len(class_names),
        num_decoder_layers=1,
        num_fusion_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        # common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        common_heads=dict(center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[0, -25.6, -3.0, 51.2, 25.6, 2.0],
            score_threshold=0.0,
            code_size=8,
            # code_size=10,
        ),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=1.0),
        # loss_iou=dict(type='CrossEntropyLoss', use_sigmoid=True, reduction='mean', loss_weight=0.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),
    train_cfg=dict(
        pts=dict(
            dataset='VODDataset',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
                iou_cost=dict(type='IoU3DCost', weight=0.25)
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            # grid_size=[1504, 1504, 40],
            grid_size= [512, 512, 40], # [1504, 1504, 40], #[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),
    test_cfg=dict(
        pts=dict(
            dataset='VODDataset',
            # grid_size=[1504, 1504, 40],
            grid_size=[512, 512, 40], #[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[0:2],
            voxel_size=voxel_size[:2],
            nms_type=None,
        )))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)  # for 8gpu * 2sample_per_gpu, #0.0001
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(
    policy='cyclic',
    target_ratio=(10, 0.0001),
    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)
total_epochs = 20
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = "/home/xuanyu/radarfusion/TransFusion/models/bevfusion_small_e4_r50.pth"
# load_from = "models/bevfusion_formal40_trained_C.pth"
# "work_dirs/bevtransfusion_vod_voxel_L/epoch_4.pth"
# "models/bevfusion_model_r50.pth" "models/transfusionL_fade_e18.pth" 'checkpoints/fusion_voxel0075_R50.pth', "models/bevfusion_fade_e18_retrained.pth"
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)
freeze_lidar_components = True
find_unused_parameters = True
