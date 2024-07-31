# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import json

from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
from ..fcn_autoencoder.unet import unetUp



@ALGORITHMS.register_module()
class MY_MOCO_V6(BaseModel):
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
                 head=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 **kwargs):
        super(MY_MOCO_V6, self).__init__(init_cfg)
        assert neck is not None
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

        self.neck_q = self.encoder_q[1]
        self.neck_k = self.encoder_k[1]
        assert head is not None
        self.head = build_head(head)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue
        self.register_buffer('queue', torch.randn(feat_dim, queue_len))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """Momentum update of the key encoder."""
        for param_q, param_k in zip(self.encoder_q.parameters(),
                                    self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + \
                           param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """Update queue."""
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_len % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.transpose(0, 1)
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

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

    def forward_train(self, img, label, **kwargs):
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
        q_features = [self.backbone(im_q)[-1]]  # index-0 means the stage 3
        q = q_features[0]
        q = self.neck_q([q])[0]  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # update the key encoder
            self._momentum_update_key_encoder()

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)

            k_features = [self.backbone_k(im_k)[-1]]  # index-0 means the stage 3
            k = k_features[0]
            k = self.neck_k([k])[0]  # queries: NxC
            k = nn.functional.normalize(k, dim=1)

            # undo shuffle
            k = batch_unshuffle_ddp(k, idx_unshuffle)

        # begin to computer loss
        losses = dict()
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        ## log l_pos_mean
        l_pos_mean = l_pos.mean(dim=0)
        l_neg_mean = l_neg.mean()
        l_neg_var = l_neg.var()
        l_mix = l_pos_mean / l_neg_mean
        l_pos_mean = float(l_pos_mean)
        l_neg_mean = float(l_neg_mean)
        l_neg_var = float(l_neg_var)
        l_mix = float(l_mix)
        ##########
        loss_constrative = self.head(l_pos, l_neg)['loss']
        losses['loss_constrative'] = loss_constrative * 0.05  # accroding to area  0.082

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
        losses['loss_reconstruction'] = rec_loss(src_area, re_area)

        self._dequeue_and_enqueue(k)

        ### json log
        filename = 'l_mean.json'
        data = {'l_pos_mean': str(l_pos_mean), 'l_neg_mean': str(l_neg_mean), 'l_neg_var': str(l_neg_var),
                'l_mix': str(l_mix)}
        with open(filename, 'w') as file_obj:
            json.dump(data, file_obj)

        return losses
