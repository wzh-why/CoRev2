# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import torch
import torch.nn as nn
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel


@ALGORITHMS.register_module()
class CoRev2(BaseModel):
    """SimMIM.

    Implementation of `SimMIM: A Simple Framework for Masked Image Modeling
    <https://arxiv.org/abs/2111.09886>`_.

    Args:
        backbone (dict): Config dict for encoder. Defaults to None.
        neck (dict): Config dict for encoder. Defaults to None.
        head (dict): Config dict for loss functions. Defaults to None.
        init_cfg (dict, optional): Config dict for weight initialization.
            Defaults to None.
    """

    def __init__(self,
                 backbone: dict,
                 neck: dict,
                 head: dict,
                 contra_neck: dict,
                 contra_head: dict,
                 base_momentum:0.99,
                 contra_weight:1.0,
                 mim_weight:1.0,
                 init_cfg: Optional[dict] = None) -> None:
        super(CoRev2, self).__init__(init_cfg)

        assert backbone is not None
        self.backbone = build_backbone(backbone)
        self.backbone_k = build_backbone(backbone)
        assert neck is not None
        self.neck = build_neck(neck)
        assert head is not None
        self.head = build_head(head)

        ## 构建对比分支
        assert contra_neck is not None
        self.contra_neck = build_neck(contra_neck)
        self.contra_neck_k = build_neck(contra_neck)

        self.base_encoder = nn.Sequential(
            self.backbone, self.contra_neck)
        self.momentum_encoder = nn.Sequential(
            self.backbone_k, self.contra_neck_k)

        assert contra_head is not None
        self.contra_head = build_head(contra_head)
        self.base_momentum = base_momentum
        self.momentum = base_momentum
        self.contra_weight = contra_weight
        self.mim_weight = mim_weight

    def init_weights(self):
        """Initialize base_encoder with init_cfg defined in backbone."""
        super(CoRev2, self).init_weights()

        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data.copy_(param_b.data)
            param_m.requires_grad = False

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the momentum encoder."""
        for param_b, param_m in zip(self.base_encoder.parameters(),
                                    self.momentum_encoder.parameters()):
            param_m.data = param_m.data * self.momentum + param_b.data * (
                1. - self.momentum)

    def extract_feat(self, img: torch.Tensor) -> tuple:
        """Function to extract features from backbone.

        Args:
            img (torch.Tensor): Input images of shape (N, C, H, W).

        Returns:
            tuple[Tensor]: Latent representations of images.
        """
        return self.backbone(img)

    def forward_train(self, x: List[torch.Tensor], **kwargs) -> dict:
        """Forward the masked image and get the reconstruction loss.

        Args:
            x (List[torch.Tensor, torch.Tensor,tuple]): view1,view2,tuple(img_rec,mask).

        Returns:
            dict: Reconstructed loss.
        """

        ### 对比分支，先实现单层
        view1,view2 = x[:2]
        view1 = view1.cuda(non_blocking=True)
        view2 = view2.cuda(non_blocking=True)
        # compute query features, [N, C] each
        q1 = self.backbone(view1,mask=None) ##取最后一层输出，得到的是一个tuple，其channels =1024，size为 (32,1024,7,7)
        q1 = self.contra_neck([q1[0]])[0]  ##要求输入是个list

        q2 = self.backbone(view2, mask=None)  ##取最后一层输出，得到的是一个tuple，其channels =1024，size为 (32,1024,7,7)
        q2 = self.contra_neck([q2[0]])[0]

        # compute key features, [N, C] each, no gradient
        with torch.no_grad():
            # here we use hook to update momentum encoder, which is a little
            # bit different with the official version but it has negligible
            # influence on the results
            k1 = self.backbone_k(view1, mask=None)
            k1 = self.contra_neck_k([k1[0]])[0]

            k2 = self.backbone_k(view2, mask=None)
            k2 = self.contra_neck_k([k2[0]])[0]

        losses = dict()
        losses['loss_contra'] = self.contra_weight * (self.contra_head(q1, k2)['loss'] + self.contra_head(q2, k1)['loss'])

        # mim task 分支
        img, mask = x[-1]
        img_latent = self.backbone(img, mask)
        img_rec = self.neck(img_latent[0])
        losses['loss_rec'] = self.mim_weight * (self.head(img, img_rec, mask)['loss'])

        return losses
