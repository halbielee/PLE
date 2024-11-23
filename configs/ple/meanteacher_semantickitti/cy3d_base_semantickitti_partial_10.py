_base_ = [
    '../_base_/datasets/semantickitti_partial_10.py',
    '../_base_/schedules/schedule-3x.py',
    '../_base_/models/cylinder3d.py',
    '../_base_/default_runtime.py',
]

randomness = dict(
    seed=1205,
    deterministic=False,
    diff_rank_seed=True)

# model settings
grid_shape = [240, 180, 20]
model = dict(
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
        type='Cylinder3DHead',
        channels=128,
        num_classes=20,
        loss_ce=dict(
            type='mmdet.CrossEntropyLoss',
            use_sigmoid=False,
            class_weight=None,
            loss_weight=1.0),
        loss_lovasz=dict(
            type='LovaszLoss',
            loss_weight=3.0,
            reduction='none')),
    train_cfg=None,
    test_cfg=dict(mode='whole'))

train_dataloader = _base_.train_dataloader
train_dataloader.batch_size = 4

# optimizer
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
