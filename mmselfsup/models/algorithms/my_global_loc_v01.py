# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import json
from torchvision.transforms import transforms
from mmselfsup.utils import (batch_shuffle_ddp, batch_unshuffle_ddp,
                             concat_all_gather)
from ..builder import ALGORITHMS, build_backbone, build_head, build_neck
from .base import BaseModel
from ..fcn_autoencoder.unet import unetUp
from .align_featuremap import align_featuremap,random_choice,UnNormalize

@ALGORITHMS.register_module()
class MY_MoCo_V91(BaseModel):
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
                 head_global=None,
                 head_1oc=None,
                 queue_len=65536,
                 feat_dim=128,
                 momentum=0.999,
                 init_cfg=None,
                 **kwargs):
        super(MY_MoCo_V91, self).__init__(init_cfg)
        self.encoder_q = nn.Sequential(
            build_backbone(backbone), build_neck(neck),
        )
        self.encoder_k = nn.Sequential(
            build_backbone(backbone), build_neck(neck),
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

        self.neck_q = self.encoder_q[1]
        self.neck_k = self.encoder_k[1]

        self.head_gl = build_head(head_global)
        self.head_loc = build_head(head_1oc)

        self.queue_len = queue_len
        self.momentum = momentum

        # create the queue for jigsaw
        self.register_buffer("queue_global", torch.randn(feat_dim, queue_len))
        self._buffers["queue_global"] = \
            nn.functional.normalize(self._buffers["queue_global"], dim=0)
        self.register_buffer("queue_ptr_global", torch.zeros(1, dtype=torch.long))

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

    def forward_train(self, img, mask, **kwargs):
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
        im_loc_tensor = img[2]
        im_loc, order = random_choice(im_loc_tensor)  # 这里的操作是让 batch_size个图像都经过同样的裁剪
        img4 = img[3]  ## image inpainting
        batch_size = im_q.size(0)
        img4_mask = mask.permute(0,3,1,2)  ### 遮挡区域的mask

        q_features = [self.backbone(im_q)[-1]]  # index-0 means the stage 3
        q_global = q_features[0]  ### for global  contanstive

        ### begin to obtain loc feature vector
        q_1 = q_features[0][:, :, :2, :2]
        q_2 = q_features[0][:, :, :2, 2:7]
        q_3 = q_features[0][:, :, 2:7, :2]
        q_4 = q_features[0][:, :, 2:7, 2:7]

        q_global = self.neck_q([q_global])[0]  # queries: NxC
        q_global = nn.functional.normalize(q_global, dim=1)  # normalize

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
            im_loc, idx_unshuffle_loc = batch_shuffle_ddp(im_loc)  ## 将图像分给不同的GPU

            k_global = [self.backbone_k(im_k)[-1]][0]  # index-0 means the stage 3
            k_loc = [self.backbone_k(im_loc)[-1]]  # index-0 means the stage 3
            k_1,k_2,k_3,k_4 = align_featuremap(k_loc,order)  ### 直接通过align_featuremap 函数进行硬匹配 此时的 k_i 直接对应于 q_i

            ### normalize
            k_global = self.neck_k([k_global])[0]  # queries: NxC
            k_global = nn.functional.normalize(k_global, dim=1)  # normalize

            k_1 = self.neck_k([k_1])[0]  # queries: NxC
            k_1 = nn.functional.normalize(k_1, dim=1)  # normalize

            k_2 = self.neck_k([k_2])[0]  # queries: NxC
            k_2 = nn.functional.normalize(k_2, dim=1)  # normalize

            k_3 = self.neck_k([k_3])[0]  # queries: NxC
            k_3 = nn.functional.normalize(k_3, dim=1)  # normalize

            k_4 = self.neck_k([k_4])[0]  # queries: NxC
            k_4 = nn.functional.normalize(k_4, dim=1)  # normalize

            ###
            k_global = batch_unshuffle_ddp(k_global, idx_unshuffle)
            k_1 = batch_unshuffle_ddp(k_1, idx_unshuffle_loc)
            k_2 = batch_unshuffle_ddp(k_2, idx_unshuffle_loc)
            k_3 = batch_unshuffle_ddp(k_3, idx_unshuffle_loc)
            k_4 = batch_unshuffle_ddp(k_4, idx_unshuffle_loc)

        # begin to computer loss
        losses = dict()
        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        ###  montagea global contrastive loss
        l_pos_gl = torch.einsum('nc,nc->n', [q_global, k_global]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_gl = torch.einsum('nc,ck->nk', [q_global, self._buffers["queue_global"].clone().detach()])
        ## log l_pos_mean
        l_pos_gl_mean = l_pos_gl.mean(dim=0)
        l_neg_gl_mean = l_neg_gl.mean()
        l_neg_gl_var = l_neg_gl.var()
        l_mix_gl = l_pos_gl_mean / l_neg_gl_mean
        l_pos_gl_mean = float(l_pos_gl_mean)
        l_neg_gl_mean = float(l_neg_gl_mean)
        l_neg_gl_var = float(l_neg_gl_var)
        l_mix_gl = float(l_mix_gl)
        ##########
        loss_constrative = self.head_gl(l_pos_gl, l_neg_gl)['loss']
        losses['loss_constrative_global'] = loss_constrative  # accroding to area  0.082

        ### loc loss
        l_pos_1 = torch.einsum('nc,nc->n', [q_1, k_1]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_1 = torch.einsum('nc,ck->nk', [q_1, self._buffers["queue_1"].clone().detach()])
        l_pos_1_mean = l_pos_1.mean(dim=0)
        l_neg_1_mean = l_neg_1.mean()
        l_neg_1_var = l_neg_1.var()
        l_mix_1 = l_pos_1_mean / l_neg_1_mean
        l_pos_1_mean = float(l_pos_1_mean)
        l_neg_1_mean = float(l_neg_1_mean)
        l_neg_1_var = float(l_neg_1_var)
        l_mix_1 = float(l_mix_1)
        loss_1 = self.head_loc(l_pos_1, l_neg_1)['loss']  # return dict, but we need to get loss
        losses['loss_contrastive_loc_1'] = loss_1 * 0.25  # accroding to area  0.082

        l_pos_2 = torch.einsum('nc,nc->n', [q_2, k_2]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_2 = torch.einsum('nc,ck->nk', [q_2, self._buffers["queue_2"].clone().detach()])
        l_pos_2_mean = l_pos_2.mean(dim=0)
        l_neg_2_mean = l_neg_2.mean()
        l_neg_2_var = l_neg_2.var()
        l_mix_2 = l_pos_2_mean / l_neg_2_mean
        l_pos_2_mean = float(l_pos_2_mean)
        l_neg_2_mean = float(l_neg_2_mean)
        l_neg_2_var = float(l_neg_2_var)
        l_mix_2 = float(l_mix_2)
        ### select
        loss_2 = self.head_loc(l_pos_2, l_neg_2)['loss']
        losses['loss_contrastive_loc_2'] = loss_2 * 0.25  # accroding to area 0.204

        l_pos_3 = torch.einsum('nc,nc->n', [q_3, k_3]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_3 = torch.einsum('nc,ck->nk', [q_3, self._buffers["queue_3"].clone().detach()])
        l_pos_3_mean = l_pos_3.mean(dim=0)
        l_neg_3_mean = l_neg_3.mean()
        l_neg_3_var = l_neg_3.var()
        l_mix_3 = l_pos_3_mean / l_neg_3_mean
        l_pos_3_mean = float(l_pos_3_mean)
        l_neg_3_mean = float(l_neg_3_mean)
        l_neg_3_var = float(l_neg_3_var)
        l_mix_3 = float(l_mix_3)
        loss_3 = self.head_loc(l_pos_3, l_neg_3)['loss']
        losses['loss_contrastive_loc_3'] = loss_3 * 0.25  # accroding to area 0.204

        l_pos_4 = torch.einsum('nc,nc->n', [q_4, k_4]).unsqueeze(-1)
        # negative logits: NxK
        l_neg_4 = torch.einsum('nc,ck->nk', [q_4, self._buffers["queue_4"].clone().detach()])
        l_pos_4_mean = l_pos_4.mean(dim=0)
        l_neg_4_mean = l_neg_4.mean()
        l_neg_4_var = l_neg_4.var()
        l_mix_4 = l_pos_4_mean / l_neg_4_mean
        l_pos_4_mean = float(l_pos_4_mean)
        l_neg_4_mean = float(l_neg_4_mean)
        l_neg_4_var = float(l_neg_4_var)
        l_mix_4 = float(l_mix_4)
        loss_4 = self.head_loc(l_pos_4, l_neg_4)['loss']
        losses['loss_contrastive_loc_4'] = loss_4 * 0.25  # accroding to area  0.510

        ####  begin to reconstruction
        [feat5, feat4, feat3, feat2, feat1] = self.backbone(img4)[::-1] ## obtain all backbone outputs
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
        src_area = img4_mask * im_q  ## [16,3,224,224] *dot product im_q [16,3,224,224]
        re_area = img4_mask * re_img  ## only need cutouted area
        losses['loss_reconstruction'] = rec_loss(src_area, re_area) / (batch_size)

        # 利用一个 Batch 的负样本更新队列：
        self._dequeue_and_enqueue(k_global, self._buffers["queue_global"], self._buffers["queue_ptr_global"])
        self._dequeue_and_enqueue(k_1, self._buffers["queue_1"], self._buffers["queue_ptr_1"])
        self._dequeue_and_enqueue(k_2, self._buffers["queue_2"], self._buffers["queue_ptr_2"])
        self._dequeue_and_enqueue(k_3, self._buffers["queue_3"], self._buffers["queue_ptr_3"])
        self._dequeue_and_enqueue(k_4, self._buffers["queue_4"], self._buffers["queue_ptr_4"])

        ### json log
        filename = 'l_mean_gl.json'
        data = {'l_pos_mean': str(l_pos_gl_mean), 'l_neg_mean': str(l_neg_gl_mean), 'l_neg_var': str(l_neg_gl_var),
                'l_mix_gl': str(l_mix_gl),
                'l_pos_1_mean': str(l_pos_1_mean), 'l_neg_1_mean': str(l_neg_1_mean), 'l_neg_1_var': str(l_neg_1_var),
                'l_mix_1': str(l_mix_1),
                'l_pos_2_mean': str(l_pos_2_mean), 'l_neg_2_mean': str(l_neg_2_mean), 'l_neg_2_var': str(l_neg_2_var),
                'l_mix_2': str(l_mix_2),
                'l_pos_3_mean': str(l_pos_3_mean), 'l_neg_3_mean': str(l_neg_3_mean), 'l_neg_3_var': str(l_neg_3_var),
                'l_mix_3': str(l_mix_3),
                'l_pos_4_mean': str(l_pos_4_mean), 'l_neg_4_mean': str(l_neg_4_mean), 'l_neg_4_var': str(l_neg_4_var),
                'l_mix_4': str(l_mix_4),
                }
        with open(filename, 'a+') as file_obj:
            json.dump(data, file_obj)
            file_obj.write('\n')

        return losses


