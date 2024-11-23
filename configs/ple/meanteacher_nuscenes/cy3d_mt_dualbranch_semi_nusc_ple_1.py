_base_ = [
    './cy3d_mt_dualbranch_semi_nusc_ple_0.5.py',
]

# quota
labeled_dataset = _base_.labeled_dataset
labeled_dataset.ann_file = 'nuscenes_kitti_infos_train.ple.1.pkl'

unlabeled_dataset = _base_.unlabeled_dataset
unlabeled_dataset.ann_file='nuscenes_kitti_infos_train.ple.1-unlabeled.pkl'

train_dataloader = dict(
    batch_size=8, num_workers=8, persistent_workers=True,
    sampler=dict(
        type='mmdet.MultiSourceSampler', batch_size=8, source_ratio=[1, 1],
    ),
    dataset=dict(
        type='ConcatDataset', datasets=[labeled_dataset, unlabeled_dataset],
    )
)

val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator