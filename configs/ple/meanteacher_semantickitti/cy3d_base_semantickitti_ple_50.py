_base_ = [
    './cy3d_base_semantickitti_partial_10.py',
]

# quota
train_dataloader = _base_.train_dataloader
train_dataloader.dataset.ann_file = 'semantickitti_infos_train.ple.50.pkl'


val_evaluator = dict(type='SegMetric')
test_evaluator = val_evaluator