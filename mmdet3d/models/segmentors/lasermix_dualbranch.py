# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mmdet.models.utils import rename_loss_dict, reweight_loss_dict

from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample, PointData
from mmdet3d.structures.det3d_data_sample import SampleList, List
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig
from .semi_base import SemiBase3DSegmentor


@MODELS.register_module()
class LaserMixDualBranch(SemiBase3DSegmentor):

    """
    LaserMix 3D segmentor with dual branch architecture.
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
        mse_dual (bool, optional): Whether to use dual branch for mean squared error loss. Default is False.
    """
    def __init__(self,
                 segmentor_student: ConfigType,
                 segmentor_teacher: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 loss_mse: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 src_pseudo_mask: str = 'strong',
                 mse_dual: bool = False) -> None:
        super(LaserMixDualBranch, self).__init__(
            segmentor_student=segmentor_student,
            segmentor_teacher=segmentor_teacher,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)

        self.src_pseudo_mask = src_pseudo_mask
        self.mse_dual = mse_dual

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

        # 2. generate pseudo labels for unlabeled data
        logits_unsup_t, pseudo_data_samples = self.get_pseudo_instances(
            multi_batch_inputs['unsup'], multi_batch_data_samples['unsup'])

        # 3. mt loss - prob of student and teacher should be similar
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

            if self.mse_dual:
                logits_s = torch.cat([logits_sup_s['weak_logits'], logits_unsup_s['weak_logits']])
                logits_t = torch.cat([logits_sup_t['weak_logits'], logits_unsup_t['weak_logits']])

                logits_s = F.softmax(logits_s, dim=1)
                logits_t = F.softmax(logits_t, dim=1)

                losses['loss_mt_dual'] = self.loss_mse(logits_s, logits_t.detach())

        # 4-1. mix loss - mix labeled and unlabeled data
        mix_batch_points = []
        mix_data_samples = []
        mix_data_idx = []

        for frame in range(len(multi_batch_inputs['sup']['points'])):
            data_sample_mix1 = Det3DDataSample()
            data_sample_mix2 = Det3DDataSample()
            pts_seg_mix1 = PointData()
            pts_seg_mix2 = PointData()
            labels = multi_batch_data_samples['sup'][
                frame].gt_pts_seg.pts_semantic_mask
            pseudo_labels = pseudo_data_samples[
                frame].gt_pts_seg.pts_semantic_mask

            points_mix1, points_mix2, labels_mix1, labels_mix2, data1_idx, data2_idx = \
                self.laser_mix_transform(
                    points_sup=multi_batch_inputs['sup']['points'][frame],
                    points_unsup=multi_batch_inputs['unsup']['points'][frame],
                    labels=labels,
                    pseudo_labels=pseudo_labels)

            mix_batch_points.append(points_mix1)
            mix_batch_points.append(points_mix2)

            pts_seg_mix1['pts_semantic_mask'] = labels_mix1
            pts_seg_mix2['pts_semantic_mask'] = labels_mix2
            data_sample_mix1.gt_pts_seg = pts_seg_mix1
            data_sample_mix2.gt_pts_seg = pts_seg_mix2

            mix_data_samples.append(data_sample_mix1)
            mix_data_samples.append(data_sample_mix2)

            mix_data_idx.append(data1_idx)
            mix_data_idx.append(data2_idx)

        mix_data = dict(
            inputs=dict(points=mix_batch_points),
            data_samples=mix_data_samples)
        mix_data = self.student.data_preprocessor(mix_data, training=True)

        # 4-2. mix loss - mix labeled and unlabeled data
        logits_mix_s, losses_mix = self.loss_by_mix_instances(
            mix_data['inputs'], mix_data['data_samples'], mix_data_idx)
        losses.update(**losses_mix)

        return losses

    def loss_by_gt_instances(
            self, batch_inputs: dict,
            batch_data_samples: SampleList) -> Tuple[Tensor, dict]:
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
        logits = self.student(batch_inputs, batch_data_samples, mode='tensor')
        losses = self.student.decode_head.loss_by_feat(logits,
                                                        batch_data_samples)
        sup_weight = self.semi_train_cfg.get('sup_weight', 1.)
        losses = rename_loss_dict('sup_',
                                  reweight_loss_dict(losses, sup_weight))
        return logits, losses

    @torch.no_grad()
    def get_pseudo_instances(
            self, batch_inputs: Dict[str, dict],
            batch_data_samples: SampleList) -> Tuple[Tensor, SampleList]:
        """Get pseudo instances from teacher model."""
        logits = self.teacher(batch_inputs, batch_data_samples, mode='tensor')
        results_list = self.teacher.decode_head.get_logits(
            logits, batch_data_samples, self.src_pseudo_mask)

        for data_samples, results in zip(batch_data_samples, results_list):
            seg_logits = F.softmax(results, dim=1)

            seg_scores, seg_labels = seg_logits.max(dim=1)
            pseudo_thr = self.semi_train_cfg.get('pseudo_thr', 0.)
            ignore_mask = (seg_scores < pseudo_thr)
            seg_labels[ignore_mask] = self.semi_train_cfg.ignore_label
            data_samples.set_data(
                {'gt_pts_seg': PointData(**{'pts_semantic_mask': seg_labels})})
        return logits, batch_data_samples

    def loss_by_mix_instances(self, batch_inputs: dict,
                              batch_data_samples: SampleList,
                              data_idx: List) -> dict:

        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (dict): Input sample dict which includes 'points' and
                'imgs' keys.

                - points (List[Tensor]): Point cloud of each sample.
                - imgs (Tensor, optional): Image tensor has shape (B, C, H, W).
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`.
            data_idx (Tensor): The index of each area in the mixed point cloud

        Returns:
            dict: A dictionary of loss components
        """
        logits = self.student(batch_inputs, batch_data_samples, mode='tensor')
        losses = self.student.decode_head.loss_by_feat_dual(logits,
                                                            batch_data_samples,
                                                            data_idx)
        unsup_weight = self.semi_train_cfg.get('unsup_weight', 1.)
        losses = rename_loss_dict('unsup_',
                                  reweight_loss_dict(losses, unsup_weight))
        return logits, losses

    def laser_mix_transform(
            self, points_sup: Tensor, points_unsup: Tensor, labels: Tensor,
            pseudo_labels: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:

        pitch_angle_down = self.semi_train_cfg.pitch_angles[0] / 180 * np.pi
        pitch_angle_up = self.semi_train_cfg.pitch_angles[1] / 180 * np.pi

        rho_sup = torch.sqrt(points_sup[:, 0] ** 2 + points_sup[:, 1] ** 2)
        pitch_sup = torch.atan2(points_sup[:, 2], rho_sup)
        pitch_sup = torch.clamp(pitch_sup, pitch_angle_down + 1e-5,
                                pitch_angle_up - 1e-5)

        rho_unsup = torch.sqrt(points_unsup[:, 0] ** 2 + points_unsup[:, 1] ** 2)
        pitch_unsup = torch.atan2(points_unsup[:, 2], rho_unsup)
        pitch_unsup = torch.clamp(pitch_unsup, pitch_angle_down + 1e-5,
                                  pitch_angle_up - 1e-5)

        num_areas = np.random.choice(self.semi_train_cfg.num_areas, size=1)[0]
        angle_list = np.linspace(pitch_angle_up, pitch_angle_down,
                                 num_areas + 1)
        points_mix1 = []
        points_mix2 = []

        labels_mix1 = []
        labels_mix2 = []

        # store the index of each area in the mixed point cloud
        data1_idx = []
        data2_idx = []

        for i in range(num_areas):
            # convert angle to radian
            start_angle = angle_list[i + 1]
            end_angle = angle_list[i]
            idx_sup = (pitch_sup > start_angle) & (pitch_sup <= end_angle)
            idx_unsup = (pitch_unsup > start_angle) & (
                    pitch_unsup <= end_angle)
            if i % 2 == 0:  # pick from original point cloud
                sup_idx = torch.full_like(labels[idx_sup], True, dtype=torch.bool)
                unsup_idx = torch.full_like(pseudo_labels[idx_unsup], False, dtype=torch.bool)
                data1_idx.append(sup_idx)
                data2_idx.append(unsup_idx)

                points_mix1.append(points_sup[idx_sup])
                labels_mix1.append(labels[idx_sup])
                points_mix2.append(points_unsup[idx_unsup])
                labels_mix2.append(pseudo_labels[idx_unsup])

            else:  # pickle from mixed point cloud
                sup_idx = torch.full_like(labels[idx_sup], True, dtype=torch.bool)
                unsup_idx = torch.full_like(pseudo_labels[idx_unsup], False, dtype=torch.bool)
                data1_idx.append(unsup_idx)
                data2_idx.append(sup_idx)
                points_mix1.append(points_unsup[idx_unsup])
                labels_mix1.append(pseudo_labels[idx_unsup])
                points_mix2.append(points_sup[idx_sup])
                labels_mix2.append(labels[idx_sup])

        points_mix1 = torch.cat(points_mix1)
        points_mix2 = torch.cat(points_mix2)
        labels_mix1 = torch.cat(labels_mix1)
        labels_mix2 = torch.cat(labels_mix2)

        data1_idx = torch.cat(data1_idx)
        data2_idx = torch.cat(data2_idx)

        return points_mix1, points_mix2, labels_mix1, labels_mix2, data1_idx, data2_idx