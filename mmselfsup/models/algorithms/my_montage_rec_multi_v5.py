# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
from ..fcn_autoencoder.unet import unetUp


@ALGORITHMS.register_module()
class MY_MoCo_v5(BaseModel):
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
                 head_1=None,
                 head_2=None,
                 head_3=None,
                 head_4=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 use_stages=[32, 16, 8, 4],
                 **kwargs):
        super(MY_MoCo_v5, self).__init__(init_cfg)
        assert neck_c5 is not None

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

        self.head_1 = build_head(head_1)
        self.head_2 = build_head(head_2)
        self.head_3 = build_head(head_3)
        self.head_4 = build_head(head_4)

        self.queue_len = queue_len
        self.momentum = momentum

        self.use_stages = use_stages

        ## create 4 * 4= 16 quene for
        for stage in self.use_stages:
            # create the queue for subfigure_1
            self.register_buffer("queue_1_{}x".format(stage), torch.randn(feat_dim, queue_len))
            self._buffers["queue_1_{}x".format(stage)] = \
                nn.functional.normalize(self._buffers["queue_1_{}x".format(stage)], dim=0)
            self.register_buffer("queue_ptr_1_{}x".format(stage), torch.zeros(1, dtype=torch.long))
            # create the queue for subfigure_2
            self.register_buffer("queue_2_{}x".format(stage), torch.randn(feat_dim, queue_len))
            self._buffers["queue_2_{}x".format(stage)] = \
                nn.functional.normalize(self._buffers["queue_2_{}x".format(stage)], dim=0)
            self.register_buffer("queue_ptr_2_{}x".format(stage), torch.zeros(1, dtype=torch.long))
            # create the queue for subfigure_3
            self.register_buffer("queue_3_{}x".format(stage), torch.randn(feat_dim, queue_len))
            self._buffers["queue_3_{}x".format(stage)] = \
                nn.functional.normalize(self._buffers["queue_3_{}x".format(stage)], dim=0)
            self.register_buffer("queue_ptr_3_{}x".format(stage), torch.zeros(1, dtype=torch.long))
            # create the queue for subfigure_4
            self.register_buffer("queue_4_{}x".format(stage), torch.randn(feat_dim, queue_len))
            self._buffers["queue_4_{}x".format(stage)] = \
                nn.functional.normalize(self._buffers["queue_4_{}x".format(stage)], dim=0)
            self.register_buffer("queue_ptr_4_{}x".format(stage), torch.zeros(1, dtype=torch.long))
        # 把对应的层 用字典构建
        self.build_dict()

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

        im_q = img[0]  ## src
        im_k = img[1]  ## montage
        img3 = img[2]  ## image inpainting

        print('img_size',im_q.shape)

        ##  do not add ,  else dict ->tuple

        q_1_list = dict()
        q_2_list = dict()
        q_3_list = dict()
        q_4_list = dict()

        q_feats = self.backbone(im_q)[::-1]
        print('q_feats _size', q_feats[0].shape)
        for idx, stage in enumerate(self.use_stages):
            q_stage = q_feats[idx]
            neck_stage_q = self.netq_dict['neck_{}x'.format(stage)]
            q_stage_1 = q_stage[:, :, :2, :2]
            q_stage_2 = q_stage[:, :, :2, 2:7]
            q_stage_3 = q_stage[:, :, 2:7, :2]
            q_stage_4 = q_stage[:, :, 2:7, 2:7]

            q_stage_1 = neck_stage_q([q_stage_1])[0]
            q_stage_1 = nn.functional.normalize(q_stage_1, dim=1)  # normalize

            q_1_list['q_{}_1'.format(stage)] = q_stage_1

            q_stage_2 = neck_stage_q([q_stage_2])[0]
            q_stage_2 = nn.functional.normalize(q_stage_2, dim=1)  # normalize
            q_2_list['q_{}_2'.format(stage)] = q_stage_2

            q_stage_3 = neck_stage_q([q_stage_3])[0]
            q_stage_3 = nn.functional.normalize(q_stage_3, dim=1)  # normalize
            q_3_list['q_{}_3'.format(stage)] = q_stage_3

            q_stage_4 = neck_stage_q([q_stage_4])[0]
            q_stage_4 = nn.functional.normalize(q_stage_4, dim=1)  # normalize
            q_4_list['q_{}_4'.format(stage)] = q_stage_4

        # compute key features
        with torch.no_grad():  # no gradient to keys

            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            # 先做shuffleBN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k_1_list = {}
            k_2_list = {}
            k_3_list = {}
            k_4_list = {}

            k_feats = self.backbone_k(im_k)[::-1]
            for idx, stage in enumerate(self.use_stages):
                k_stage = k_feats[idx]
                neck_stage_k = self.netk_dict['neck_{}x'.format(stage)]

                k_stage_1 = k_stage[:, :, 5:7, 5:7]
                k_stage_2 = k_stage[:, :, 5:7, :5]
                k_stage_3 = k_stage[:, :, :5, 5:7]
                k_stage_4 = k_stage[:, :, :5, :5]

                k_stage_1 = neck_stage_k([k_stage_1])[0]
                k_stage_1 = nn.functional.normalize(k_stage_1, dim=1)  # normalize
                # 再做 Unshuffle
                k_stage_1 = batch_unshuffle_ddp(k_stage_1, idx_unshuffle)
                k_1_list['k_{}_1'.format(stage)] = k_stage_1

                k_stage_2 = neck_stage_k([k_stage_2])[0]
                k_stage_2 = nn.functional.normalize(k_stage_2, dim=1)  # normalize
                # 再做 Unshuffle
                k_stage_2 = batch_unshuffle_ddp(k_stage_2, idx_unshuffle)
                k_2_list['k_{}_2'.format(stage)] = k_stage_2

                k_stage_3 = neck_stage_k([k_stage_3])[0]
                k_stage_3 = nn.functional.normalize(k_stage_3, dim=1)  # normalize
                # 再做 Unshuffle
                k_stage_3 = batch_unshuffle_ddp(k_stage_3, idx_unshuffle)
                k_3_list['k_{}_3'.format(stage)] = k_stage_3

                k_stage_4 = neck_stage_k([k_stage_4])[0]
                k_stage_4 = nn.functional.normalize(k_stage_4, dim=1)  # normalize
                # 再做 Unshuffle
                k_stage_4 = batch_unshuffle_ddp(k_stage_4, idx_unshuffle)
                k_4_list['k_{}_4'.format(stage)] = k_stage_4

            # compute key features

        # begin to computer loss
        losses = dict()
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1

        ## in the patch order
        ## firse to compute q_list_1 and k_list_1
        loss_1_total = 0
        for idx, stage in enumerate(self.use_stages):
            q_1_stage = q_1_list['q_{}_1'.format(stage)]
            k_1_stage = k_1_list['k_{}_1'.format(stage)]
            l_pos_1_stage = torch.einsum('nc,nc->n', [q_1_stage, k_1_stage]).unsqueeze(-1)
            l_neg_1_stage = torch.einsum('nc,ck->nk', [q_1_stage, self._buffers["queue_1_{}x".format(stage)].clone().detach()])
            l_loss_1_stage = self.head_1(l_pos_1_stage, l_neg_1_stage)['loss']  # return dict, but we need to get loss
            loss_1_total += l_loss_1_stage
        losses['loss_montage_1'] = loss_1_total * 0.25  # accroding to area  0.082

        loss_2_total = 0
        for idx, stage in enumerate(self.use_stages):
            q_2_stage = q_2_list['q_{}_2'.format(stage)]
            k_2_stage = k_2_list['k_{}_2'.format(stage)]
            l_pos_2_stage = torch.einsum('nc,nc->n', [q_2_stage, k_2_stage]).unsqueeze(-1)
            l_neg_2_stage = torch.einsum('nc,ck->nk',
                                         [q_2_stage, self._buffers["queue_2_{}x".format(stage)].clone().detach()])
            l_loss_2_stage = self.head_2(l_pos_2_stage, l_neg_2_stage)['loss']  # return dict, but we need to get loss
            loss_2_total += l_loss_2_stage
        losses['loss_montage_2'] = loss_2_total * 0.25  # accroding to area  0.082

        loss_3_total = 0
        for idx, stage in enumerate(self.use_stages):
            q_3_stage = q_3_list['q_{}_3'.format(stage)]
            k_3_stage = k_3_list['k_{}_3'.format(stage)]
            l_pos_3_stage = torch.einsum('nc,nc->n', [q_3_stage, k_3_stage]).unsqueeze(-1)
            l_neg_3_stage = torch.einsum('nc,ck->nk',
                                         [q_3_stage, self._buffers["queue_3_{}x".format(stage)].clone().detach()])
            l_loss_3_stage = self.head_3(l_pos_3_stage, l_neg_3_stage)['loss']  # return dict, but we need to get loss
            loss_3_total += l_loss_3_stage
        losses['loss_montage_3'] = loss_3_total * 0.25  # accroding to area  0.082

        loss_4_total = 0
        for idx, stage in enumerate(self.use_stages):
            q_4_stage = q_4_list['q_{}_4'.format(stage)]
            k_4_stage = k_4_list['k_{}_4'.format(stage)]
            l_pos_4_stage = torch.einsum('nc,nc->n', [q_4_stage, k_4_stage]).unsqueeze(-1)
            l_neg_4_stage = torch.einsum('nc,ck->nk',
                                         [q_4_stage, self._buffers["queue_4_{}x".format(stage)].clone().detach()])
            l_loss_4_stage = self.head_4(l_pos_4_stage, l_neg_4_stage)['loss']  # return dict, but we need to get loss
            loss_4_total += l_loss_4_stage
        losses['loss_montage_4'] = loss_4_total * 0.25  # accroding to area  0.082

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

        batch_img_number = im_q.shape[0]
        src_area = im_q[:, :, 80:144, 80:144]
        re_area = re_img[:, :, 80:144, 80:144]

        losses['loss_reconstruction'] = 0.9 * rec_loss(src_area, re_area)

        # 利用一个 Batch 的负样本更新队列：
        for idx, stage in enumerate(self.use_stages):
            k_1_st = k_1_list['k_{}_1'.format(stage)]
            k_2_st = k_2_list['k_{}_2'.format(stage)]
            k_3_st = k_3_list['k_{}_3'.format(stage)]
            k_4_st = k_4_list['k_{}_4'.format(stage)]

            # 更新queue_global
            self._dequeue_and_enqueue(k_1_st,
                                      self._buffers['queue_1_{}x'.format(stage)],
                                      self._buffers['queue_ptr_1_{}x'.format(stage)])

            self._dequeue_and_enqueue(k_2_st,
                                      self._buffers['queue_2_{}x'.format(stage)],
                                      self._buffers['queue_ptr_2_{}x'.format(stage)])

            self._dequeue_and_enqueue(k_3_st,
                                      self._buffers['queue_3_{}x'.format(stage)],
                                      self._buffers['queue_ptr_3_{}x'.format(stage)])
            self._dequeue_and_enqueue(k_4_st,
                                      self._buffers['queue_4_{}x'.format(stage)],
                                      self._buffers['queue_ptr_4_{}x'.format(stage)])

        return losses
