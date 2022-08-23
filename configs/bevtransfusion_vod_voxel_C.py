point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2]
class_names = ['Car', 'Pedestrian', 'Cyclist']
voxel_size = [0.1, 0.1, 0.125]
out_size_factor = 8
evaluation = dict(interval=1)
dataset_type = 'VODDataset'
data_root = 'data/view_of_delft_PUBLIC/radar'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
img_scale = (448, 800) #TODO wrong
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

numC_Trans=256

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
    # dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True), #for debug
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
            dict(type='MyResize', img_scale=img_scale, keep_ratio=True),
            dict(type='MyNormalize', **img_norm_cfg),
            dict(type='MyPad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                # with_label=False
                ),
            dict(type='Collect3D', keys=['points', 'img'])  
            # dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=0,
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
    # pretrained=dict(
    #     img="models/nuScenes_3Ddetection_e140.pth"
    # ),
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
        out_indices=(1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),    
    img_neck=dict(
        type='GeneralizedLSSFPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,   
        start_level=0,
        num_outs=3,
        norm_cfg=dict(type="BN2d",requires_grad=True),
        act_cfg=dict(type="ReLU",inplace=True),
        upsample_cfg=dict(mode="bilinear",align_corners=False)),
    img_vtransform=dict(
        type="DepthLSSTransform", #"DepthLSSTransform",
        in_channels=256,
        out_channels=256, #original 80
        image_size=img_scale,
        feature_size= (img_scale[0]//8,img_scale[1]//8),
        xbound=[0, 51.2, 0.4],
        ybound=[-25.6, 25.6, 0.4],
        zbound=[-3.0, 2.0, 5.0], #[-10,10,20]
        dbound=[1.0, 60.0, 0.5], 
        downsample=2), #[51.2/0.4/2] = 64
    img_bev_encoder_backbone=dict(
        type='ResNetForBEVDet',
        numC_input=numC_Trans
    ),
    img_bev_encoder_neck=dict(
        type='FPN_LSS',
        in_channels=numC_Trans*8+numC_Trans*2,
        out_channels=256
    ),
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=[class_names[0]]),
            dict(num_class=1, class_names=[class_names[1]]),
            dict(num_class=1, class_names=[class_names[2]]),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],
            out_size_factor=out_size_factor,
            post_center_range=[0, -25.6, -3.0, 51.2, 25.6, 2.0],
            score_threshold=0.0,
            code_size=8,
            max_num=500),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=10.0),
        norm_bbox=True),
    train_cfg=dict(
        pts=dict(
            dataset='VODDataset',
            # assigner=dict(
            #     type='HungarianAssigner3D',
            #     iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
            #     cls_cost=dict(type='FocalLossCost', gamma=2, alpha=0.25, weight=0.15),
            #     reg_cost=dict(type='BBoxBEVL1Cost', weight=0.25),
            #     iou_cost=dict(type='IoU3DCost', weight=0.25)
            # ),
            # pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size=[512, 512, 40], #[1440, 1440, 40],  # [x_len, y_len, 1]
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            # code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range,
            dense_reg=1,
            max_objs=500,
            )),
    test_cfg=dict(
        pts=dict(
            dataset='VODDataset',
            grid_size=[512, 512, 40], #[1440, 1440, 40],
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range[:2],
            voxel_size=voxel_size[:2],

            post_center_limit_range=point_cloud_range,
            max_per_img=300,
            max_pool_nms=False,
            min_radius=[2],
            score_threshold=0,
            pre_max_size=1000, #??
            post_max_size=15,
            # nms_type = "circle"
            # # Scale-NMS
            nms_type='rotate',
            nms_thr=[0.2, 0.5, 0.2],
            nms_rescale_factor=[1.0, 4.5, 1.0]
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
total_epochs = 80
checkpoint_config = dict(interval=1)
log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None
load_from = "work_dirs/bevtransfusion_vod_voxel_C/500obj_bboxweight2/epoch_14.pth" #"models/transfusionL_fade_e18.pth" #'checkpoints/fusion_voxel0075_R50.pth'
resume_from = None
workflow = [('train', 1)]
gpu_ids = range(0, 8)
find_unused_parameters = True
