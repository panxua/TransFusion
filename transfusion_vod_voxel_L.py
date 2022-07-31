point_cloud_range = [-75.2, -75.2, -2, 75.2, 75.2, 4]
class_names = ['Car', 'Pedestrian', 'Cyclist']
voxel_size = [0.1, 0.1, 0.15]
out_size_factor = 8
evaluation = dict(interval=1)
dataset_type = 'VODDataset'
data_root = 'data/view_of_delft_PUBLIC/radar'
input_modality = dict(
    use_lidar=False,
    use_camera=False,
    use_radar=True,
    use_map=False,
    use_external=False)
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3]),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05]),
    dict(
        type='PointsRangeFilter',
        point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]),
    dict(
        type='ObjectRangeFilter',
        point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]),
    dict(type='PointShuffle'),
    dict(
        type='DefaultFormatBundle3D',
        class_names=['Car', 'Pedestrian', 'Cyclist']),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=7,
        use_dim=[0, 1, 2, 3]),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(800, 1333),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter',
                point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]),
            dict(
                type='DefaultFormatBundle3D',
                class_names=['Car', 'Pedestrian', 'Cyclist'],
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=0,
    train=dict(
        type='RepeatDataset',
        times=1,
        dataset=dict(
            type='VODDataset',
            data_root='data/view_of_delft_PUBLIC/radar',
            ann_file=
            'data/view_of_delft_PUBLIC/radar/vod_radar_infos_train.pkl',
            split='training',
            pipeline=[
                dict(
                    type='LoadPointsFromFile',
                    coord_type='LIDAR',
                    load_dim=7,
                    use_dim=[0, 1, 2, 3]),
                dict(
                    type='LoadAnnotations3D',
                    with_bbox_3d=True,
                    with_label_3d=True),
                dict(
                    type='RandomFlip3D',
                    sync_2d=False,
                    flip_ratio_bev_horizontal=0.5,
                    flip_ratio_bev_vertical=0.5),
                dict(
                    type='GlobalRotScaleTrans',
                    rot_range=[-0.78539816, 0.78539816],
                    scale_ratio_range=[0.95, 1.05]),
                dict(
                    type='PointsRangeFilter',
                    point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]),
                dict(
                    type='ObjectRangeFilter',
                    point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]),
                dict(type='PointShuffle'),
                dict(
                    type='DefaultFormatBundle3D',
                    class_names=['Car', 'Pedestrian', 'Cyclist']),
                dict(
                    type='Collect3D',
                    keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
            ],
            classes=['Car', 'Pedestrian', 'Cyclist'],
            modality=dict(
                use_lidar=False,
                use_camera=False,
                use_radar=True,
                use_map=False,
                use_external=False),
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type='VODDataset',
        data_root='data/view_of_delft_PUBLIC/radar',
        ann_file='data/view_of_delft_PUBLIC/radar/vod_radar_infos_val.pkl',
        split='training',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=7,
                use_dim=[0, 1, 2, 3]),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(800, 1333),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Car', 'Pedestrian', 'Cyclist'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist'],
        modality=dict(
            use_lidar=False,
            use_camera=False,
            use_radar=True,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'),
    test=dict(
        type='VODDataset',
        data_root='data/view_of_delft_PUBLIC/radar',
        ann_file='data/view_of_delft_PUBLIC/radar/vod_radar_infos_val.pkl',
        split='training',
        pipeline=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=7,
                use_dim=[0, 1, 2, 3]),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(800, 1333),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1.0, 1.0],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter',
                        point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=['Car', 'Pedestrian', 'Cyclist'],
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ],
        classes=['Car', 'Pedestrian', 'Cyclist'],
        modality=dict(
            use_lidar=False,
            use_camera=False,
            use_radar=True,
            use_map=False,
            use_external=False),
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='TransFusionDetector',
    pts_voxel_layer=dict(
        max_num_points=5,
        voxel_size=[0.1, 0.1, 0.15],
        max_voxels=150000,
        point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]),
    pts_voxel_encoder=dict(
        type='HardVFE',
        in_channels=4,
        feat_channels=[64],
        with_distance=False,
        with_cluster_center=False,
        with_voxel_center=False,
        voxel_size=[0.1, 0.1, 0.15],
        norm_cfg=dict(type='BN1d', eps=0.001, momentum=0.01),
        point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4]),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=64,
        sparse_shape=[41, 1504, 1504],
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32), (32, 32, 64), (64, 64, 128), (128,
                                                                      128)),
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
        num_proposals=300,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=3,
        num_decoder_layers=1,
        num_heads=8,
        learnable_query_pos=False,
        initialize_by_heatmap=True,
        nms_kernel_size=3,
        ffn_channel=256,
        dropout=0.1,
        bn_momentum=0.1,
        activation='relu',
        common_heads=dict(
            center=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        bbox_coder=dict(
            type='TransFusionBBoxCoder',
            pc_range=[-75.2, -75.2],
            voxel_size=[0.1, 0.1],
            out_size_factor=8,
            post_center_range=[-80, -80, -10.0, 80, 80, 10.0],
            score_threshold=0.0,
            code_size=8),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=2.0),
        loss_heatmap=dict(
            type='GaussianFocalLoss', reduction='mean', loss_weight=1.0)),
    train_cfg=dict(
        pts=dict(
            dataset='VODDataset',
            assigner=dict(
                type='HungarianAssigner3D',
                iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
                cls_cost=dict(
                    type='FocalLossCost', gamma=2, alpha=0.25, weight=0.6),
                reg_cost=dict(type='BBoxBEVL1Cost', weight=2.0),
                iou_cost=dict(type='IoU3DCost', weight=2.0)),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[1504, 1504, 40],
            voxel_size=[0.1, 0.1, 0.15],
            out_size_factor=8,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            point_cloud_range=[-75.2, -75.2, -2, 75.2, 75.2, 4])),
    test_cfg=dict(
        pts=dict(
            dataset='VODDataset',
            grid_size=[1504, 1504, 40],
            out_size_factor=8,
            voxel_size=[0.1, 0.1],
            nms_type=None)))
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.01)
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
total_epochs = 36
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = '.'
load_from = None
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 2)
