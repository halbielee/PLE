# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, OrderedDict, Tuple

import torch
import torch.nn.functional as F
from mmdet.models.utils import rename_loss_dict, reweight_loss_dict
from torch import Tensor, nn

from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.structures.det3d_data_sample import OptSampleList, SampleList
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .semi_base import SemiBase3DSegmentor


@MODELS.register_module()
class MeanTeacherDualBranch3DSegmentor(SemiBase3DSegmentor):
    """
    MeanTeacher 3D segmentor with dual branch architecture.
    This is proposed in 'Learning from Spatio-temporal Correlation for Semi-Supervised LiDAR Semantic Segmentation'
    (https://arxiv.org/abs/2410.06893).

    Args:
        segmentor_student (ConfigType): Configuration for the student segmentor.
        segmentor_teacher (ConfigType): Configuration for the teacher segmentor.
        semi_train_cfg (OptConfigType, optional): Configuration for semi-supervised training.
        semi_test_cfg (OptConfigType, optional): Configuration for semi-supervised testing.
        loss_mse (OptConfigType, optional): Configuration for mean squared error loss.
        data_preprocessor (OptConfigType, optional): Configuration for data preprocessing.
        init_cfg (OptMultiConfig, optional): Initialization configuration.
        src_pseudo_mask (str, optional): Source pseudo mask type. Default is "strong".
    """
    def __init__(
        self,
        segmentor_student: ConfigType,
        segmentor_teacher: ConfigType,
        semi_train_cfg: OptConfigType = None,
        semi_test_cfg: OptConfigType = None,
        loss_mse: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
        init_cfg: OptMultiConfig = None,
        src_pseudo_mask: str = "strong",
    ) -> None:
        super(MeanTeacherDualBranch3DSegmentor, self).__init__(
            segmentor_student=segmentor_student,
            segmentor_teacher=segmentor_teacher,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.src_pseudo_mask = src_pseudo_mask
        if loss_mse is not None:
            self.loss_mse = MODELS.build(loss_mse)
        else:
            self.loss_mse = None

    def loss(
        self,
        multi_batch_inputs: Dict[str, dict],
        multi_batch_data_samples: Dict[str, SampleList],
    ) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        losses = dict()

        # 1. supervised loss for labeled data
        logits_sup_s, losses_sup = self.loss_by_gt_instances(
            multi_batch_inputs["sup"], multi_batch_data_samples["sup"]
        )
        losses.update(**losses_sup)

        # 2-1. get pseudo labels for unlabeled data
        logits_unsup_t, pseudo_data_samples = self.get_pseudo_instances(
            multi_batch_inputs["unsup"], multi_batch_data_samples["unsup"]
        )

        # 2-2. unsupervised loss for unlabeled data
        logits_unsup_s, losses_unsup = self.loss_by_pseudo_instances(
            multi_batch_inputs["unsup"], pseudo_data_samples
        )
        losses.update(**losses_unsup)

        # 2. mt loss - prob of student and teacher should be similar
        if self.loss_mse is not None:
            logits_sup_t = self.teacher(
                multi_batch_inputs["sup"],
                multi_batch_data_samples["sup"],
                mode="tensor",
            )
            logits_unsup_s = self.student(
                multi_batch_inputs["unsup"],
                multi_batch_data_samples["unsup"],
                mode="tensor",
            )

            logits_s = torch.cat([logits_sup_s["logits"], logits_unsup_s["logits"]])
            logits_t = torch.cat([logits_sup_t["logits"], logits_unsup_t["logits"]])

            logits_s = F.softmax(logits_s, dim=1)
            logits_t = F.softmax(logits_t, dim=1)

            losses["loss_mt"] = self.loss_mse(logits_s, logits_t.detach())

        return losses

    def loss_by_gt_instances(
        self, batch_inputs: dict, batch_data_samples: SampleList
    ) -> Tuple[Tensor, dict]:
        """Calculate losses from a batch of inputs and ground-truth data
        samples.

        Args:
            batch_inputs (dict): Input sample dict which includes 'points' and
                'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            Tuple[Tensor, dict]: Predict logits and a dictionary of loss
            components.
        """
        logits = self.student(batch_inputs, batch_data_samples, mode="tensor")
        losses = self.student.decode_head.loss_by_feat(logits, batch_data_samples)
        sup_weight = self.semi_train_cfg.get("sup_weight", 1.0)
        losses = rename_loss_dict("sup_", reweight_loss_dict(losses, sup_weight))
        return logits, losses

    @torch.no_grad()
    def get_pseudo_instances(
        self, batch_inputs: Dict[str, dict], batch_data_samples: SampleList
    ) -> Tuple[Tensor, SampleList]:
        """Get pseudo instances from teacher model."""
        logits = self.teacher(batch_inputs, batch_data_samples, mode="tensor")
        results_list = self.teacher.decode_head.get_logits(
            logits, batch_data_samples, self.src_pseudo_mask
        )

        for data_samples, results in zip(batch_data_samples, results_list):
            seg_logits = F.softmax(results, dim=1)

            seg_scores, seg_labels = seg_logits.max(dim=1)
            pseudo_thr = self.semi_train_cfg.get("pseudo_thr", 0.0)
            ignore_mask = seg_scores < pseudo_thr
            seg_labels[ignore_mask] = self.semi_train_cfg.ignore_label
            data_samples.set_data(
                {"gt_pts_seg": PointData(**{"pts_semantic_mask": seg_labels})}
            )
        return logits, batch_data_samples

    def loss_by_pseudo_instances(
        self, batch_inputs: dict, batch_data_samples: SampleList
    ) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (dict): Input sample dict which includes 'points' and
                'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.

        Returns:
            dict: A dictionary of loss components
        """
        logits = self.student(batch_inputs, batch_data_samples, mode="tensor")

        losses = self.student.decode_head.loss_by_feat_weak(logits, batch_data_samples)
        unsup_weight = self.semi_train_cfg.get("unsup_weight", 1.0)
        losses = rename_loss_dict("unsup_", reweight_loss_dict(losses, unsup_weight))
        return logits, losses
