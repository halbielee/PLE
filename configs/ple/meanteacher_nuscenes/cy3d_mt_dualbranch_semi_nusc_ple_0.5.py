_base_ = [
    '../_base_/datasets/semi_nusc_kitti-seg.py',
    '../_base_/schedules/schedule-3x.py',
    '../_base_/default_runtime.py',
]

dataset_type = 'NuScenesSegDataset'

data_root = 'data/nuscenes_kitti/'
class_names = [
    'noise', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
    'vegetation'
]
labels_map = {
    1: 0, 5: 0, 7: 0, 8: 0, 10: 0, 11: 0, 13: 0, 19: 0, 20: 0, 0: 0, 29: 0, 31: 0,
    9: 1,
    14: 2,
    15: 3, 16: 3,
    17: 4,
    18: 5,
    21: 6,
    2: 7, 3: 7, 4: 7, 6: 7,
    12: 8,
    22: 9,
    23: 10,
    24: 11,
    25: 12,
    26: 13,
    27: 14,
    28: 15,
    30: 16
}

metainfo = dict(
    classes=class_names, seg_label_mapping=labels_map, max_label=31)


input_modality = dict(use_lidar=True, use_camera=False)

data_prefix = dict(
    pts='',
    pts_semantic_mask='')

backend_args = None
branch_field = ['sup', 'unsup']

randomness = dict(
    seed=1205,
    deterministic=False,
    diff_rank_seed=True)

# pipeline used to augment labeled data,
# which will be sent to student model for supervised training.
sup_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        with_attr_label=False,
        seg_3d_dtype='np.uint8',
        backend_args=backend_args),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(
        type='MultiBranch3D',
        branch_field=branch_field,
        sup=dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask']))
]
# pipeline used to augment unlabeled data,
# which will be sent to teacher model for predicting pseudo instances.

unsup_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=4,
        backend_args=backend_args),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.78539816, 0.78539816],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0.1, 0.1, 0.1],
    ),
    dict(
        type='MultiBranch3D',
        branch_field=branch_field,
        unsup=dict(
            type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask']))
]

grid_shape = [240, 180, 20]

segmentor = dict(
    type='Cylinder3D',
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True, voxel_type='cylindrical',
        voxel_layer=dict(
            grid_shape=grid_shape,
            point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
            max_num_points=-1,
            max_voxels=-1)),
    voxel_encoder=dict(
        type='SegVFE',
        feat_channels=[64, 128, 256, 256],
        in_channels=6,
        with_voxel_center=True,
        feat_compression=16,
        grid_shape=grid_shape),
    backbone=dict(
        type='Asymm3DSpconv',
        grid_size=grid_shape,
        input_channels=16,
        base_channels=32,
        norm_cfg=dict(type='BN1d', eps=1e-5, momentum=0.1)),
    decode_head=dict(
        type='CylinderDualBranch3DHead',
        channels=128,
        num_classes=17,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        loss_lovasz=dict(
            type='LovaszLoss',
            loss_weight=3.0,
            reduction='none'),
        weak_internal_dim=512,
        ignore_index=0),
    train_cfg=None,
    test_cfg=dict(mode='whole'))

model = dict(
    type='MeanTeacherDualBranch3DSegmentor',
    segmentor_student=segmentor,
    segmentor_teacher=segmentor,
    data_preprocessor=dict(
        type='MultiBranch3DDataPreprocessor',
        data_preprocessor=dict(
            type='Det3DDataPreprocessor',
            voxel=True,
            voxel_type='cylindrical',
            voxel_layer=dict(
                grid_shape=grid_shape,
                point_cloud_range=[0, -3.14159265359, -4, 50, 3.14159265359, 2],
                max_num_points=-1, max_voxels=-1))),
    loss_mse=(dict(
        type='mmdet.MSELoss',
        loss_weight=250)),
    semi_train_cfg=dict(
        freeze_teacher=True,
        pseudo_thr=0.90,
        ignore_label=0,
        pitch_angles=[-25, 3],
        num_areas=[4, 5, 6, 7, 8],
        sup_weight=1,
        unsup_weight=1),
    semi_test_cfg=dict(
        extract_feat_on='teacher',
        predict_on='teacher'),
    src_pseudo_mask='strong')

# quota
# quota
labeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=data_prefix,
    pipeline=sup_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    ignore_index=0,
    test_mode=False,
    backend_args=backend_args,
    ann_file='nuscenes_kitti_infos_train.ple.0.5.pkl'
)
unlabeled_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    data_prefix=data_prefix,
    pipeline=unsup_pipeline,
    metainfo=metainfo,
    modality=input_modality,
    ignore_index=0,
    test_mode=False,
    backend_args=backend_args,
    ann_file='nuscenes_kitti_infos_train.ple.0.5.pkl',
)
train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='mmdet.MultiSourceSampler',
                 batch_size=8,
                 source_ratio=[1, 1]),
    dataset=dict(type='ConcatDataset',
                 datasets=[labeled_dataset, unlabeled_dataset]))

# learning rate
lr = 0.008
optim_wrapper = dict(type='OptimWrapper',
                     optimizer=dict(type='AdamW',
                                    lr=lr,
                                    weight_decay=0.01),
                     clip_grad=dict(max_norm=10,
                                    norm_type=2))

param_scheduler = [dict(type='OneCycleLR',
                        total_steps=45000,  # 45000 iters for 8xb2 or 4xb4; 23000 iters for 8xb4 or 4xb8
                        by_epoch=False, eta_max=0.001)]

train_cfg = dict(_delete_=True,
                 type='IterBasedTrainLoop',
                 max_iters=45000,  # 45000 iters for 8xb2 or 4xb4; 23000 iters for 8xb4 or 4xb8
                 val_interval=1000)

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator

# default hook
default_hooks = dict(checkpoint=dict(
    by_epoch=False,
    save_best='miou',
    rule='greater'))
log_processor = dict(by_epoch=False)

custom_hooks = [dict(type='mmdet.MeanTeacherHook',
                     momentum=0.01)]
