# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.runner import BaseModule

from mmselfsup.utils import concat_all_gather
from ..builder import HEADS, build_neck


@HEADS.register_module()
class MoCoV3Head(BaseModule):
    """Head for MoCo v3 algorithms.

    This head builds a predictor, which can be any registered neck component.
    It also implements latent contrastive loss between two forward features.
    Part of the code is modified from:
    `<https://github.com/facebookresearch/moco-v3/blob/main/moco/builder.py>`_.

    Args:
        predictor (dict): Config dict for module of predictor.
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Defaults to 1.0.
    """

    def __init__(self, predictor, temperature=1.0):
        super(MoCoV3Head, self).__init__()
        self.predictor = build_neck(predictor)
        self.temperature = temperature

    def forward(self, base_out, momentum_out):
        """Forward head.

        Args:
            base_out (Tensor): NxC features from base_encoder.
            momentum_out (Tensor): NxC features from momentum_encoder.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        # predictor computation
        pred = self.predictor([base_out])[0]

        # normalize
        pred = nn.functional.normalize(pred, dim=1)
        target = nn.functional.normalize(momentum_out, dim=1) ##本身就是非对称设计

        # get negative samples
        # target = concat_all_gather(target)  ##合并所有keys，适合于分布式，将其注释掉使其适合单机版本

        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [pred, target]) / self.temperature

        # generate labels
        batch_size = logits.shape[0]

        # labels = (torch.arange(batch_size, dtype=torch.long) +
        #           batch_size * torch.distributed.get_rank()).cuda()  ## 创建多卡版的标签，默认对角线上是正确的

        labels = torch.arange(batch_size, dtype=torch.long).cuda()  ## 创建单卡版的标签，默认对角线上是正确的

        loss = 2 * self.temperature * nn.CrossEntropyLoss()(logits, labels)
        return dict(loss=loss)
