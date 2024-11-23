# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .semi_base import SemiBase3DSegmentor


@MODELS.register_module()
class MeanTeacher3DSegmentor(SemiBase3DSegmentor):
    """Base class for semi-supervised segmentors.

    Semi-supervised segmentors typically consisting of a teacher model updated
    by exponential moving average and a student model updated by gradient
    descent.

    Args:
        segmentor_student (:obj:`ConfigDict` or dict): The segmentor-student config.
        segmentor_teacher (:obj:`ConfigDict` or dict): The segmentor-teacher config.
        semi_train_cfg (:obj:`ConfigDict` or dict, optional): The
            semi-supervised training config. Defaults to None.
        semi_test_cfg (:obj:`ConfigDict` or dict, optional): The semi-segmentor
            testing config. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`Det3DDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict or List[:obj:`ConfigDict` or dict],
            optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 segmentor_student: ConfigType,
                 segmentor_teacher: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 loss_mse: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super(MeanTeacher3DSegmentor, self).__init__(
            segmentor_student=segmentor_student,
            segmentor_teacher=segmentor_teacher,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        if loss_mse is not None:
            self.loss_mse = MODELS.build(loss_mse)
        else:
            self.loss_mse = None

    def loss(self, multi_batch_inputs: Dict[str, dict],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from a batch of inputs and data samples."""

        losses = dict()

        # 1. supervised loss for labeled data
        logits_sup_s, losses_sup = self.loss_by_gt_instances(
            multi_batch_inputs['sup'], multi_batch_data_samples['sup'])
        losses.update(**losses_sup)

        # generate pseudo labels for unlabeled data
        logits_unsup_t, pseudo_data_samples = self.get_pseudo_instances(
            multi_batch_inputs['unsup'], multi_batch_data_samples['unsup'])

        # 2. mt loss - prob of student and teacher should be similar
        if self.loss_mse is not None:
            logits_sup_t = self.teacher(multi_batch_inputs['sup'],
                                        multi_batch_data_samples['sup'],
                                        mode='tensor')
            logits_unsup_s = self.student(multi_batch_inputs['unsup'],
                                          multi_batch_data_samples['unsup'],
                                          mode='tensor')

            logits_s = torch.cat([logits_sup_s['logits'], logits_unsup_s['logits']])
            logits_t = torch.cat([logits_sup_t['logits'], logits_unsup_t['logits']])

            logits_s = F.softmax(logits_s, dim=1)
            logits_t = F.softmax(logits_t, dim=1)

            losses['loss_mt'] = self.loss_mse(logits_s, logits_t.detach())

        # 3. unsupervised loss for unlabeled data
        logits_unsup_t, pseudo_data_samples = self.get_pseudo_instances(
            multi_batch_inputs['unsup'], multi_batch_data_samples['unsup'])
        logits_unsup_s, losses_unsup = self.loss_by_pseudo_instances(
            multi_batch_inputs['unsup'], pseudo_data_samples)
        losses.update(**losses_unsup)
        return losses
