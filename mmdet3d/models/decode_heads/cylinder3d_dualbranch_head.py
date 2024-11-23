# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn.functional as F
from mmcv.ops import SparseModule, SubMConv3d, Conv2d
from torch import Tensor

from mmdet3d.models.data_preprocessors.voxelize import dynamic_scatter_3d
from mmdet3d.registry import MODELS
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils import OptMultiConfig
from mmdet3d.utils.typing_utils import ConfigType
from .cylinder3d_head import Cylinder3DHead


@MODELS.register_module()
class CylinderDualBranch3DHead(Cylinder3DHead):

    """
    Cylinder3DHead with dual branch
    This is proposed in 'Learning from Spatio-temporal Correlation for Semi-Supervised LiDAR Semantic Segmentation'
    (https://arxiv.org/abs/2410.06893).

    Args:
    - `channels` (int): The number of input channels.
    - `num_classes` (int): The number of output classes for segmentation.
    - `dropout_ratio` (float, optional): The dropout ratio. Default is 0.
    - `conv_cfg` (ConfigType, optional): Configuration for convolution layers. Default is `dict(type='Conv1d')`.
    - `norm_cfg` (ConfigType, optional): Configuration for normalization layers. Default is `dict(type='BN1d')`.
    - `act_cfg` (ConfigType, optional): Configuration for activation layers. Default is `dict(type='ReLU')`.
    - `loss_ce` (ConfigType, optional): Configuration for the cross-entropy loss. Default is `dict(type='mmdet.CrossEntropyLoss', use_sigmoid=False, class_weight=None, loss_weight=1.0)`.
    - `loss_lovasz` (ConfigType, optional): Configuration for the Lovasz loss. Default is `dict(type='LovaszLoss', loss_weight=1.0)`.
    - `conv_seg_kernel_size` (int, optional): The kernel size for the segmentation convolution layer. Default is 3.
    - `ignore_index` (int, optional): The label index to ignore during loss calculation. Default is 19.
    - `init_cfg` (OptMultiConfig, optional): Initialization configuration. Default is None.
    - `weak_internal_dim` (int, optional): The internal dimension for the weak branch. Default is 512.
    """

    def __init__(self,
                 channels: int,
                 num_classes: int,
                 dropout_ratio: float = 0,
                 conv_cfg: ConfigType = dict(type='Conv1d'),
                 norm_cfg: ConfigType = dict(type='BN1d'),
                 act_cfg: ConfigType = dict(type='ReLU'),
                 loss_ce: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=False,
                     class_weight=None,
                     loss_weight=1.0),
                 loss_lovasz: ConfigType = dict(
                     type='LovaszLoss', loss_weight=1.0),
                 conv_seg_kernel_size: int = 3,
                 ignore_index: int = 19,
                 init_cfg: OptMultiConfig = None,
                 weak_internal_dim: int = 512,
                 ) -> None:
        super(CylinderDualBranch3DHead, self).__init__(
            channels=channels,
            num_classes=num_classes,
            dropout_ratio=dropout_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            loss_ce=loss_ce,
            loss_lovasz=loss_lovasz,
            conv_seg_kernel_size=conv_seg_kernel_size,
            ignore_index=ignore_index,
            init_cfg=init_cfg)

        self.weak_extractor, self.weak_conv_seg = None, None
        self.build_weak_branch(channels,
                               num_classes,
                               conv_seg_kernel_size,
                               weak_internal_dim)

    def build_weak_branch(self, channels,
                          num_classes, conv_seg_kernel_size, weak_internal_dim):
        self.weak_conv_seg = SubMConv3d(
            channels,
            num_classes,
            indice_key='weak_logit',
            kernel_size=conv_seg_kernel_size,
            stride=1,
            padding=1,
            bias=True)

    def weak_cls_seg(self, feat: Tensor) -> Tensor:
        """Classify each points."""
        if self.dropout is not None:
            feat = self.dropout(feat)

        if self.weak_extractor:
            output = self.weak_extractor(feat).features
            output = self.weak_conv_seg(output[:, :, None, None]).squeeze(-1).squeeze(-1)
        else:
            output = self.weak_conv_seg(feat)
            output = output.features

        return output

    def forward(self, voxel_dict: dict) -> dict:
        """Forward function."""
        sparse_logits = self.cls_seg(voxel_dict['voxel_feats'])
        voxel_dict['logits'] = sparse_logits.features

        weak_sparse_logits = self.weak_cls_seg(voxel_dict['voxel_feats'])
        voxel_dict['weak_logits'] = weak_sparse_logits

        return voxel_dict

    def get_logits(self, voxel_dict: dict,
                   batch_data_samples: SampleList,
                   src_pseudo_mask: str) -> List[Tensor]:
        """Predict function.

        Args:
            voxel_dict (dict): The dict may contain `sparse_logits`,
                `point2voxel_map`.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The det3d data
                samples. It usually includes information such as `metainfo` and
                `gt_pts_seg`. We use `point2voxel_map` in this function.
            src_pseudo_mask (str): Type of pseudo mask for weak branch. Defaults to 'strong',

        Returns:
            List[Tensor]: List of point-wise segmentation logits.
        """

        if src_pseudo_mask == 'strong':
            seg_logits = voxel_dict['logits']
        elif src_pseudo_mask == 'weak':
            seg_logits = voxel_dict['weak_logits']
        else:
            raise NotImplementedError

        seg_pred_list = []
        coors = voxel_dict['voxel_coors']
        for batch_idx in range(len(batch_data_samples)):
            batch_mask = coors[:, 0] == batch_idx
            seg_logits_sample = seg_logits[batch_mask]
            point2voxel_map = voxel_dict['point2voxel_maps'][batch_idx].long()
            point_seg_predicts = seg_logits_sample[point2voxel_map]
            seg_pred_list.append(point_seg_predicts)

        return seg_pred_list

    def loss_by_feat_weak(self, voxel_dict: dict,
                    batch_data_samples: SampleList) -> dict:
        """Compute semantic segmentation loss.

        Args:
            voxel_dict (dict): The dict may contain `sparse_logits`,
                `point2voxel_map`.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        voxel_semantic_segs = []
        coors = voxel_dict['coors']
        for batch_idx, data_sample in enumerate(batch_data_samples):
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            batch_mask = coors[:, 0] == batch_idx
            this_coors = coors[batch_mask, 1:]
            voxel_semantic_mask, _, _ = dynamic_scatter_3d(
                F.one_hot(pts_semantic_mask.long()).float(), this_coors,
                'mean')
            voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
            voxel_semantic_segs.append(voxel_semantic_mask)
        seg_label = torch.cat(voxel_semantic_segs)
        seg_logit_feat = voxel_dict['weak_logits']
        loss = dict()
        loss['loss_ce'] = self.loss_ce(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)
        loss['loss_lovasz'] = self.loss_lovasz(
            seg_logit_feat, seg_label, ignore_index=self.ignore_index)

        return loss

    def loss_by_feat_dual(self, voxel_dict: dict,
                          batch_data_samples: SampleList,
                          data_idx: Tensor) -> dict:
        """Compute semantic segmentation loss.

        Args:
            voxel_dict (dict): The dict may contain `sparse_logits`,
                `point2voxel_map`.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_pts_seg`.

        Returns:
            Dict[str, Tensor]: A dictionary of loss components.
        """

        voxel_semantic_segs = []
        voxel_indices = []
        coors = voxel_dict['coors']
        for batch_idx, data_sample in enumerate(batch_data_samples):
            pts_semantic_mask = data_sample.gt_pts_seg.pts_semantic_mask
            batch_mask = coors[:, 0] == batch_idx
            this_coors = coors[batch_mask, 1:]
            voxel_semantic_mask, _, _ = dynamic_scatter_3d(
                F.one_hot(pts_semantic_mask.long()).float(), this_coors,
                'mean')

            voxel_idx, _, _ = dynamic_scatter_3d(
                F.one_hot(data_idx[batch_idx].long()).float(), this_coors,
                'mean')

            voxel_semantic_mask = torch.argmax(voxel_semantic_mask, dim=-1)
            voxel_semantic_segs.append(voxel_semantic_mask)

            voxel_idx = torch.argmax(voxel_idx, dim=-1).bool()
            voxel_indices.append(voxel_idx)

        seg_label = torch.cat(voxel_semantic_segs)
        seg_idx = torch.cat(voxel_indices)

        strong_seg_logit_feat = voxel_dict['logits']
        weak_seg_logit_feat = voxel_dict['weak_logits']

        loss = dict()
        loss['loss_ce_s'] = self.loss_ce(
            strong_seg_logit_feat[seg_idx], seg_label[seg_idx],
            ignore_index=self.ignore_index)
        loss['loss_lovasz_s'] = self.loss_lovasz(
            strong_seg_logit_feat[seg_idx], seg_label[seg_idx],
            ignore_index=self.ignore_index)

        loss['loss_ce_w'] = self.loss_ce(
            weak_seg_logit_feat[~seg_idx], seg_label[~seg_idx],
            ignore_index=self.ignore_index)
        loss['loss_lovasz_w'] = self.loss_lovasz(
            weak_seg_logit_feat[~seg_idx], seg_label[~seg_idx],
            ignore_index=self.ignore_index)

        return loss