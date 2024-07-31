# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
from ..fcn_autoencoder.unet import unetUp


@ALGORITHMS.register_module()
class MY_MoCo_V22(BaseModel):
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
                 neck=None,
                 # neck_slic=None,
                 head_1=None,
                 head_2=None,
                 head_3=None,
                 head_4=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 **kwargs):
        super(MY_MoCo_V22, self).__init__(init_cfg)
        assert neck is not None

        ## src

        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck))

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

        self.neck_q = self.encoder_q[1]
        self.neck_k = self.encoder_k[1]

        # self.neck_slic = self.encoder_q[2],
        # add by wu to built the head for compute the loss, so it donot to  init

        self.head_1 = build_head(head_1)
        self.head_2 = build_head(head_2)
        self.head_3 = build_head(head_3)
        self.head_4 = build_head(head_4)

        self.queue_len = queue_len
        self.momentum = momentum


        # create the queue for global
        self.register_buffer("queue_1", torch.randn(feat_dim, queue_len))  ## create a constaant of (128,16636) it's value is random
        self._buffers["queue_1"] = \
            nn.functional.normalize(self._buffers["queue_1"], dim=0)
        self.register_buffer("queue_ptr_1", torch.zeros(1, dtype=torch.long))   ##  create a constaant of (1)

        # create the queue for jigsaw
        self.register_buffer("queue_2", torch.randn(feat_dim, queue_len))
        self._buffers["queue_2"] = \
            nn.functional.normalize(self._buffers["queue_2"], dim=0)
        self.register_buffer("queue_ptr_2", torch.zeros(1, dtype=torch.long))

        # create the queue for jigsaw
        self.register_buffer("queue_3", torch.randn(feat_dim, queue_len))
        self._buffers["queue_3"] = \
            nn.functional.normalize(self._buffers["queue_3"], dim=0)
        self.register_buffer("queue_ptr_3", torch.zeros(1, dtype=torch.long))

        # create the queue for jigsaw
        self.register_buffer("queue_4", torch.randn(feat_dim, queue_len))
        self._buffers["queue_4"] = \
            nn.functional.normalize(self._buffers["queue_4"], dim=0)
        self.register_buffer("queue_ptr_4", torch.zeros(1, dtype=torch.long))

        # #  add by wu
        # self.isroialign = False
        self.box = None

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

        # compute query features
        # q_features = self.encoder_k(im_q)[0]  # index-0 means the stage 3

        q_features = [self.backbone(im_q)[-1]]  # index-0 means the stage 3

        print('q_feature shape is ', q_features[0].shape)  # q_features[0] shape is [64,2048,7,7]

        q_1 = q_features[0][:, :, :2, :2]
        q_2 = q_features[0][:, :, :2, 2:7]
        q_3 = q_features[0][:, :, 2:7, :2]
        q_4 = q_features[0][:, :, 2:7, 2:7]

        q_1 = self.neck_q([q_1])[0]  # queries: NxC
        q_1 = nn.functional.normalize(q_1, dim=1)  # normalize

        q_2 = self.neck_q([q_2])[0]  # queries: NxC
        q_2 = nn.functional.normalize(q_2, dim=1)  # normalize

        q_3 = self.neck_q([q_3])[0]  # queries: NxC
        q_3 = nn.functional.normalize(q_3, dim=1)  # normalize

        q_4 = self.neck_q([q_4])[0]  # queries: NxC
        q_4 = nn.functional.normalize(q_4, dim=1)  # normalize

        # compute key features
        with torch.no_grad():  # no gradient to keys

            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k_features = [self.backbone_k(im_k)[-1]]  # index-0 means the stage 3

            k_1 = k_features[0][:, :, 5:7, 5:7]
            # k_2 = k_features[0][:, :, :5, 5:7].permute(0, 1, 3, 2)  # need to tran
            # k_3 = k_features[0][:, :, 5:7, :5].permute(0, 1, 3, 2)  # need to tran

            k_2 = k_features[0][:, :, 5:7, :5]
            k_3 = k_features[0][:, :, :5, 5:7]
            k_4 = k_features[0][:, :, :5, :5]

            k_1 = self.neck_k([k_1])[0]  # queries: NxC
            k_1 = nn.functional.normalize(k_1, dim=1)  # normalize

            k_2 = self.neck_k([k_2])[0]  # queries: NxC
            k_2 = nn.functional.normalize(k_2, dim=1)  # normalize

            k_3 = self.neck_k([k_3])[0]  # queries: NxC
            k_3 = nn.functional.normalize(k_3, dim=1)  # normalize

            k_4 = self.neck_k([k_4])[0]  # queries: NxC
            k_4 = nn.functional.normalize(k_4, dim=1)  # normalize

            # undo shuffle
            k_1 = k_1.contiguous()
            k_2 = k_2.contiguous()
            k_3 = k_3.contiguous()
            k_4 = k_4.contiguous()

            k_1 = batch_unshuffle_ddp(k_1, idx_unshuffle)
            k_2 = batch_unshuffle_ddp(k_2, idx_unshuffle)
            k_3 = batch_unshuffle_ddp(k_3, idx_unshuffle)
            k_4 = batch_unshuffle_ddp(k_4, idx_unshuffle)
            # compute key features

        # begin to computer loss
        losses = dict()
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos_1 = torch.einsum('nc,nc->n', [q_1, k_1]).unsqueeze(-1)
        l_pos_1_mean = l_pos_1.mean(dim=0)
        # negative logits: NxK
        l_neg_1 = torch.einsum('nc,ck->nk', [q_1, self._buffers["queue_1"].clone().detach()])
        loss_1 = self.head_1(l_pos_1, l_neg_1)['loss']  # return dict, but we need to get loss
        losses['loss_montage_1'] = loss_1 * 0.05  # accroding to area  0.082

        l_pos_2 = torch.einsum('nc,nc->n', [q_2, k_2]).unsqueeze(-1)
        l_pos_2_mean = l_pos_2.mean(dim=0)
        # negative logits: NxK
        l_neg_2 = torch.einsum('nc,ck->nk', [q_2, self._buffers["queue_2"].clone().detach()])
        loss_2 = self.head_2(l_pos_2, l_neg_2)['loss']
        losses['loss_montage_2'] = loss_2 * 0.05  # accroding to area 0.204

        l_pos_3 = torch.einsum('nc,nc->n', [q_3, k_3]).unsqueeze(-1)
        l_pos_3_mean = l_pos_3.mean(dim=0)
        # negative logits: NxK
        l_neg_3 = torch.einsum('nc,ck->nk', [q_3, self._buffers["queue_3"].clone().detach()])
        loss_3 = self.head_3(l_pos_3, l_neg_3)['loss']
        losses['loss_montage_3'] = loss_3 * 0.05  # accroding to area 0.204

        l_pos_4 = torch.einsum('nc,nc->n', [q_4, k_4]).unsqueeze(-1)
        l_pos_4_mean = l_pos_4.mean(dim=0)
        # negative logits: NxK
        l_neg_4 = torch.einsum('nc,ck->nk', [q_4, self._buffers["queue_4"].clone().detach()])
        loss_4 = self.head_4(l_pos_4, l_neg_4)['loss']
        losses['loss_montage_4'] = loss_4 * 0.05  # accroding to area  0.510

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

        ### open the txt to log
        file_l_pos_1 = open('l_pos_1_mean.txt', mode='a+')
        file_l_pos_2 = open('l_pos_2_mean.txt', mode='a+')
        file_l_pos_3 = open('l_pos_3_mean.txt', mode='a+')
        file_l_pos_4 = open('l_pos_4_mean.txt', mode='a+')

        ##  write l_pos  to txt
        file_l_pos_1.write('l_pos_1_mean is {} \n'.format(l_pos_1_mean))
        file_l_pos_2.write('l_pos_2_mean is {} \n'.format(l_pos_2_mean))
        file_l_pos_3.write('l_pos_3_mean is {} \n'.format(l_pos_3_mean))
        file_l_pos_4.write('l_pos_4_mean is {} \n'.format(l_pos_4_mean))

        # 利用一个 Batch 的负样本更新队列：
        self._dequeue_and_enqueue(k_1, self._buffers["queue_1"], self._buffers["queue_ptr_1"])
        self._dequeue_and_enqueue(k_2, self._buffers["queue_2"], self._buffers["queue_ptr_2"])
        self._dequeue_and_enqueue(k_3, self._buffers["queue_3"], self._buffers["queue_ptr_3"])
        self._dequeue_and_enqueue(k_4, self._buffers["queue_4"], self._buffers["queue_ptr_4"])

        return losses
