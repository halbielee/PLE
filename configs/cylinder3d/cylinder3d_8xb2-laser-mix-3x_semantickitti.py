_base_ = [
    '../_base_/datasets/semantickitti.py', '../_base_/models/cylinder3d.py',
    '../_base_/schedules/schedule-3x.py', '../_base_/default_runtime.py'
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
        prob=1),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[0., 6.28318531],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
    ),
    dict(type='Pack3DDetInputs', keys=['points', 'pts_semantic_mask'])
]

train_dataloader = dict(dataset=dict(pipeline=train_pipeline))

default_hooks = dict(checkpoint=dict(type='CheckpointHook', interval=1))
