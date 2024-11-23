_base_ = [
    '../_base_/datasets/_semantickitti_20.py',
    '../_base_/models/_cylinder3d.py',
    '../_base_/schedules/schedule-3x.py',
    '../_base_/default_runtime.py'
]

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
    dict(
        type='LoadAnnotations3D',
        with_bbox_3d=False,
        with_label_3d=False,
        with_seg_3d=True,
        seg_3d_dtype='np.int32',
        seg_offset=2**16,
        dataset_type='semantickitti'),
    dict(type='PointSegClassMapping'),
    dict(
        type='LaserMix',
        num_areas=[3, 4, 5, 6],
        pitch_angles=[-25, 3],
        pre_transform=[
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=4,
                use_dim=4),
            dict(
                type='LoadAnnotations3D',
                with_bbox_3d=False,
                with_label_3d=False,
                with_seg_3d=True,
                seg_3d_dtype='np.int32',
                seg_offset=2**16,
                dataset_type='semantickitti'),
            dict(type='PointSegClassMapping')
        ],
        prob=0.8),
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
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))
