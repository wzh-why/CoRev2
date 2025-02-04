# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class SwAV(BaseModel):
    """SwAV.

    Implementation of `Unsupervised Learning of Visual Features by Contrasting
    Cluster Assignments <https://arxiv.org/abs/2006.09882>`_.
    The queue is built in `core/hooks/swav_hook.py`.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 head=None,
                 init_cfg=None,
                 **kwargs):
        super(SwAV, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: Backbone outputs.
        """
        x = self.backbone(img)
        return x

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (list[Tensor]): A list of input images with shape
                (N, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert isinstance(img, list)
        # multi-res forward passes
        idx_crops = torch.cumsum(                  ### 返回维度dim中输入元素的累加和
            torch.unique_consecutive(              ### 返回不重复的元素
                torch.tensor([i.shape[-1] for i in img]),
                return_counts=True)[1], 0)   ## 返回索引例如，tensor[2,8]
        start_idx = 0
        output = []
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(img[start_idx:end_idx]))
            output.append(_out)
            start_idx = end_idx  ##整个output 为 两部分，0-【32，2048，7，7】 1- [96，2048，3，3]
        output = self.neck(output)[0]  ## 输出为 两个部分合在一起的 输出，即 (32+96=128(数量等于8*N),128(指定的feature 维度))

        loss = self.head(output)
        return loss
