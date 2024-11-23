import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.registry import MODELS


@MODELS.register_module()
class FocalSoftmaxLoss(nn.Module):
    def __init__(self, n_classes, gamma=1, cls_freq=0.8,
                 ignore_index=None,):
        super(FocalSoftmaxLoss, self).__init__()
        self.gamma = gamma
        self.n_classes = n_classes
        self.ignore_index = ignore_index

        # make alpha
        self.cls_freq = cls_freq
        alpha = self.calculate_alpha()
        # settings for alpha
        if isinstance(alpha, list):
            assert len(alpha) == n_classes, 'len(alpha)!=n_classes: {} vs. {}'.format(
                len(alpha), n_classes)
            self.alpha = torch.Tensor(alpha)
        elif isinstance(alpha, np.ndarray):
            assert alpha.shape[0] == n_classes, 'len(alpha)!=n_classes: {} vs. {}'.format(
                len(alpha), n_classes)
            self.alpha = torch.from_numpy(alpha)
        else:
            assert alpha < 1 and alpha > 0, 'invalid alpha: {}'.format(alpha)
            self.alpha = torch.zeros(n_classes)
            self.alpha[0] = alpha
            self.alpha[1:] += (1-alpha)

    def calculate_alpha(self):
        self.cls_freq = np.array(self.cls_freq)
        cls_weight = 1 / (self.cls_freq + 1e-8)

        alpha = np.log(1 + cls_weight)
        alpha = alpha / alpha.max()
        alpha[-1] = 0
        return alpha

    def forward(self, cls_score, label, mask=None, ignore_index=None,
                reduction_override=None,):
        """compute focal loss
        Args:
            @param cls_score: (N, C) or (N, C, H, W)
            @param label: (N) or (N, H, W)
            @param mask:
            @param ignore_index
            @param reduction_override
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        if ignore_index is None:
            ignore_index = self.ignore_index

        if cls_score.dim() > 2:
            pred = cls_score.view(cls_score.size(0), cls_score.size(1), -1)
            pred = pred.transpose(1, 2)
            pred = pred.contiguous().view(-1, cls_score.size(1))
        else:
            pred = cls_score

        # # check number of  points for each label
        # for i in range(self.n_classes):
        #     print('label {} has {} points'.format(i, (label == i).sum()))

        label = label.view(-1, 1)
        pred_softmax = F.softmax(pred, 1)
        pred_softmax = pred_softmax.gather(1, label).view(-1)

        pred_logsoft = pred_softmax.clamp(1e-6).log()
        self.alpha = self.alpha.to(cls_score.device)
        alpha = self.alpha.gather(0, label.squeeze())
        loss = - (1-pred_softmax).pow(self.gamma)
        loss = loss * pred_logsoft * alpha

        if mask is not None:
            if len(mask.size()) > 1:
                mask = mask.view(-1)
            loss = (loss * mask).sum() / mask.sum()
            return loss
        else:
            return loss.mean()


