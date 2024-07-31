# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
from ..fcn_autoencoder.unet import unetUp


@ALGORITHMS.register_module()
class MY_MOCO_V62(BaseModel):
    """MoCo.

    Implementation of `Momentum Contrast for Unsupervised Visual
    Representation Learning <https://arxiv.org/abs/1911.05722>`_.
    Part of the code is borrowed from:
    `<https://github.com/facebookresearch/moco/blob/master/moco/builder.py>`_.

    Args:
        backbone (dict): Config dict for module of backbone.
        neck (dict): Config dict for module of deep features to compact
            feature vectors. Defaults to None.
        head (dict): Config dict for module of loss functions.
            Defaults to None.
        queue_len (int): Number of negative keys maintained in the queue.
            Defaults to 65536.
        feat_dim (int): Dimension of compact feature vectors. Defaults to 128.
        momentum (float): Momentum coefficient for the momentum-updated
            encoder. Defaults to 0.999.
    """

    def __init__(self,
                 backbone,
                 neck_c5=None,
                 neck_c4=None,
                 neck_c3=None,
                 neck_c2=None,
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 use_stages=[32, 16, 8, 4],
                 **kwargs):
        super(MY_MOCO_V62, self).__init__(init_cfg)

        self.encoder_q = nn.Sequential(
            build_backbone(backbone),
            build_neck(neck_c5),
            build_neck(neck_c4),
            build_neck(neck_c3),
            build_neck(neck_c2),
        )
        self.encoder_k = nn.Sequential(
            build_backbone(backbone),
            build_neck(neck_c5),
            build_neck(neck_c4),
            build_neck(neck_c3),
            build_neck(neck_c2),
        )
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        self.backbone = self.encoder_q[0]
        self.backbone_k = self.encoder_k[0]

        self.head = build_head(head)


        self.queue_len = queue_len
        self.momentum = momentum

        self.use_stages = use_stages
        # create the queue
        for stage in self.use_stages:
            # create the queue for subfigure_1
            self.register_buffer("queue_{}x".format(stage), torch.randn(feat_dim, queue_len))
            self._buffers["queue_{}x".format(stage)] = \
                nn.functional.normalize(self._buffers["queue_{}x".format(stage)], dim=0)
            self.register_buffer("queue_ptr_{}x".format(stage), torch.zeros(1, dtype=torch.long))
            # 把对应的层 用字典构建
        self.build_dict()

        ### add the decoder
        ### decoder
        in_filters = [192, 512, 1024, 3072]
        out_filters = [64, 128, 256, 512]
        # upsampling
        # 64,64,512
        self.up_concat4 = unetUp(in_filters[3], out_filters[3])
        # 128,128,256
        self.up_concat3 = unetUp(in_filters[2], out_filters[2])
        # 256,256,128
        self.up_concat2 = unetUp(in_filters[1], out_filters[1])
        # 512,512,64
        self.up_concat1 = unetUp(in_filters[0], out_filters[0])
        self.up_conv = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(out_filters[0]),
            nn.ReLU(),
        )
        self.con_final = nn.Conv2d(out_filters[0], 3, kernel_size=1)

    def build_dict(self):
        # 定义netq_dict，其中包含 4个neck
        self.netq_dict = dict(
            neck_32x=self.encoder_q[1],
            neck_16x=self.encoder_q[2],
            neck_8x=self.encoder_q[3],
            neck_4x=self.encoder_q[4],
        )
        # 定义netk_dict，其中包含4 个neck
        self.netk_dict = dict(
            neck_32x=self.encoder_k[1],
            neck_16x=self.encoder_k[2],
            neck_8x=self.encoder_k[3],
            neck_4x=self.encoder_k[4],
        )

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue, queue_ptr):
        """Update queue."""
        keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer
        queue_ptr[0] = ptr   # change pointer

    def extract_feat(self, img):
        """Function to extract features from backbone.

        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.

        Returns:
            tuple[Tensor]: backbone outputs.
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
        im_q = img[0]
        im_k = img[1]
        img3 = img[2]  ## image inpainting

        # compute query features

        q_f_list = dict()
        q_features = self.backbone(im_q)[::-1]  ### list (c5,c4,c3,c2,c1)
        for idx, stage in enumerate(self.use_stages):
            q_stage = q_features[idx]
            neck_stage_q = self.netq_dict['neck_{}x'.format(stage)]
            q_stage_list = [q_stage]
            q_neck_f = neck_stage_q([q_stage])[0]
            q_neck_f = nn.functional.normalize(q_neck_f, dim=1)  # normalize
            q_f_list['q_{}'.format(stage)] = q_neck_f

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()
            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)
            k_f_list = dict()
            k_features = self.backbone_k(im_k)[::-1]  # index-0 means the stage 3
            for idx, stage_k in enumerate(self.use_stages):
                k_stage = k_features[idx]
                neck_stage_k = self.netk_dict['neck_{}x'.format(stage_k)]
                k_neck_f = neck_stage_k([k_stage])[0]
                k_neck_f = nn.functional.normalize(k_neck_f, dim=1)  # normalize
                # undo shuffle
                k_neck_f = batch_unshuffle_ddp(k_neck_f, idx_unshuffle)
                k_f_list['k_{}'.format(stage_k)] = k_neck_f

        # begin to computer loss
        losses = dict()
        loss_total = 0
        for idx, stage in enumerate(self.use_stages):
            q_stage = q_f_list['q_{}'.format(stage)]
            k_stage = k_f_list['k_{}'.format(stage)]
            l_pos_stage = torch.einsum('nc,nc->n', [q_stage, k_stage]).unsqueeze(-1)
            l_neg_stage = torch.einsum('nc,ck->nk',
                                         [q_stage, self._buffers["queue_{}x".format(stage)].clone().detach()])
            l_loss_stage = self.head(l_pos_stage, l_neg_stage)['loss']  # return dict, but we need to get loss
            loss_total += l_loss_stage
        losses['loss_constrative'] = loss_total * 0.25  # accroding to area  0.082

        ####  begin to reconstruction
        [feat5, feat4, feat3, feat2, feat1] = self.backbone(img3)[::-1] ## obtain all backbone outputs
        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)
        if self.up_conv != None:
            up1 = self.up_conv(up1)  # up1 shape is (img_h,img_w,64)
        ## 再添加一个卷积，使得 图片通道数为3
        re_img = self.con_final(up1)
        ## reconstruction loss
        rec_loss = nn.L1Loss()
        src_area = im_q[:, :, 80:144, 80:144]
        re_area = re_img[:, :, 80:144, 80:144]
        losses['loss_reconstruction'] = rec_loss(src_area, re_area)

        #######################
        # 利用一个 Batch 的负样本更新队列：
        for idx, stage in enumerate(self.use_stages):
            k_st = k_f_list['k_{}'.format(stage)]
            # 更新queue_global
            self._dequeue_and_enqueue(k_st,
                                      self._buffers['queue_{}x'.format(stage)],
                                      self._buffers['queue_ptr_{}x'.format(stage)])

        return losses
