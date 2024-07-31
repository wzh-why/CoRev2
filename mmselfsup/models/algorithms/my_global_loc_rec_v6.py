'''该脚本用于 在v9的基础上添加一个fpn层，意在实现架构的对齐'''

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
from .align_featuremap import align_featuremap_dist_fpn,random_choice,UnNormalize
from ..backbones.fpn import FPN

@ALGORITHMS.register_module()
class MY_MoCo_V96(BaseModel):
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
        super(MY_MoCo_V96, self).__init__(init_cfg)
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

        self.in_features = ['res1','res2','res3','res4','res5']
        outs = [] ##
        backbone_out = dict(zip(self.in_features,outs))  ### zip函数 负责打包成一个元组，即将 in_featutes 与 output 一一对应
        self.fpn_q = FPN(bottom_up=backbone_out,in_features=self.in_features)  ## c初始化
        self.fpn_k = FPN(bottom_up=backbone_out,in_features=self.in_features)  ## c初始化

        ### loc branch
        self.avapooling = nn.AdaptiveAvgPool2d((1,1))

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
        self.register_buffer("queue_loc", torch.randn(feat_dim, queue_len))  ## create a constaant of (128,16636) it's value is random
        self._buffers["queue_loc"] = \
            nn.functional.normalize(self._buffers["queue_loc"], dim=0)
        self.register_buffer("queue_ptr_loc", torch.zeros(1, dtype=torch.long))   ##  create a constaant of (1)

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

    def forward_train(self, img, order, **kwargs):
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
        im_loc = img[2]
        batch_size = im_q.size(0)
        # print('order is',order)
        img4 = img[3]  ## image inpainting
        order_list = order

        # out_list_q_gl = self.backbone(im_q)[0:5:1] ## obtain all backbone outputs 顺序获取
        backbone_out_q_gl = dict(zip(self.in_features, self.backbone(im_q)[0:5:1]))
        q_gl_fpn = [self.fpn_q(backbone_out_q_gl)['p4']]  ### 将经过fpn后的层 放入到list,并取出其中的 p4 特征图 B,256,14,14

        q_global = q_gl_fpn[0]  ### for global  contanstive
        ### begin to obtain loc feature vector
        q_1 = q_gl_fpn[0][:, :, :4, :4]
        q_2 = q_gl_fpn[0][:, :, :4, 4:14]
        q_3 = q_gl_fpn[0][:, :, 4:14, :4]
        q_4 = q_gl_fpn[0][:, :, 4:14, 4:14]

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

        tensor_q = torch.cat((q_1, q_2, q_3, q_4), dim=0)
        tensor_q_2 = tensor_q.view(-1, batch_size, 128).permute(1, 2, 0)  ### B,C,4

        # compute key features
        with torch.no_grad():  # no gradient to keys

            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            im_k, idx_unshuffle = batch_shuffle_ddp(im_k)
            im_loc, idx_unshuffle_loc = batch_shuffle_ddp(im_loc)

            # out_list_k_gl = self.backbone_k(im_q)[0:5:1]  ## obtain all backbone outputs
            backbone_k_out_gl = dict(zip(self.in_features, self.backbone_k(im_q)[0:5:1] ))
            k_global_fpn = [self.fpn_k(backbone_k_out_gl)['p4']][0]  ### 将经过fpn后的层 放入到list,并取出其中的 p4 特征图 B,256,14,14

            # out_list_k_loc = self.backbone_k(im_loc)[0:5:1]  ## obtain all backbone outputs
            backbone_k_out_loc = dict(zip(self.in_features, self.backbone_k(im_loc)[0:5:1]))
            k_loc_fpn = [self.fpn_k(backbone_k_out_loc)['p4']]  ### 将经过fpn后的层 放入到list,并取出其中的 p4 特征图 B,256,14,14 这里不需要【0】
            k_1,k_2,k_3,k_4 = align_featuremap_dist_fpn(k_loc_fpn,order_list)

            ### normalize
            k_global = self.neck_k([k_global_fpn])[0]  # queries: NxC
            k_global = nn.functional.normalize(k_global, dim=1)  # normalize

            k_1 = self.neck_k([k_1])[0]  # queries: NxC
            k_1 = nn.functional.normalize(k_1, dim=1)  # normalize

            k_2 = self.neck_k([k_2])[0]  # queries: NxC
            k_2 = nn.functional.normalize(k_2, dim=1)  # normalize

            k_3 = self.neck_k([k_3])[0]  # queries: NxC
            k_3 = nn.functional.normalize(k_3, dim=1)  # normalize

            k_4 = self.neck_k([k_4])[0]  # queries: NxC
            k_4 = nn.functional.normalize(k_4, dim=1)  # normalize

            ### similiar to densecl, we use the global pooling to B,128,2,2 TO get k_loc_update

            tensor_k = torch.cat((k_1, k_2, k_3, k_4), dim=0)
            k_loc = tensor_k.view(-1, batch_size, 128).permute(1, 2, 0)  ### B,C,4
            k_loc_pooling = k_loc.permute(2, 0, 1)  ## 4,B,C
            k_loc_pooling = k_loc_pooling.view(2, 2, batch_size, 128)  ## 2,2,B,C
            k_loc_pooling = k_loc_pooling.permute(2, 3, 0, 1)  ### 2,2,B,C -> B,C,2,2
            k_loc_pooling = self.avapooling(k_loc_pooling)  ### B,C,1,1
            k_loc_update = k_loc_pooling.view(k_loc_pooling.size(0), -1)  ### B,C

            ##
            k_global = batch_unshuffle_ddp(k_global, idx_unshuffle)
            k_loc = batch_unshuffle_ddp(k_loc, idx_unshuffle_loc)
            k_loc_update = batch_unshuffle_ddp(k_loc_update, idx_unshuffle_loc)

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
        losses['loss_constrative_global'] = loss_constrative   # accroding to area  0.082

        ### begin to compute loc loss
        sim_matrix = torch.matmul(tensor_q_2.permute(0, 2, 1), k_loc)
        dense_sim_ind = sim_matrix.max(dim=2)[1]  ###[1] denotes index [0] denotes value  obtain the match k feature vectors index
        index_k_grid = torch.gather(k_loc, 2, dense_sim_ind.unsqueeze(1).expand(-1, k_loc.size(1), -1))  ### obtain ites value
        dense_sim_q = (tensor_q_2 * index_k_grid).sum(1)  ### measure the sim between q and k

        l_pos_loc = dense_sim_q.view(-1).unsqueeze(-1)
        tensor_q_2 = tensor_q_2.permute(0, 2, 1)
        tensor_q_2 = tensor_q_2.reshape(-1, tensor_q_2.size(2))
        l_neg_loc = torch.einsum('nc,ck->nk', [tensor_q_2, self._buffers["queue_loc"].clone().detach()])

        #### log
        l_pos_loc_mean = l_pos_loc.mean(dim=0)
        l_neg_loc_mean = l_neg_loc.mean()
        l_neg_loc_var = l_neg_loc.var()
        l_mix_loc = l_pos_loc_mean / l_neg_loc_mean
        l_pos_loc_mean = float(l_pos_loc_mean)
        l_neg_loc_mean = float(l_neg_loc_mean)
        l_neg_loc_var = float(l_neg_loc_var)
        l_mix_loc = float(l_mix_loc)

        loss_constrative_loc = self.head_loc(l_pos_loc, l_neg_loc)['loss']
        losses['loss_constrative_loc'] = loss_constrative_loc  # accroding to area  0.082

        ####  begin to reconstruction
        ### begain to compute rec branch loss
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
        src_area = im_q[:, :, 80:144, 80:144]
        re_area = re_img[:, :, 80:144, 80:144]
        losses['loss_reconstruction'] = 0.9 * rec_loss(src_area, re_area)

        # 利用一个 Batch 的负样本更新队列：
        self._dequeue_and_enqueue(k_global, self._buffers["queue_global"], self._buffers["queue_ptr_global"])
        self._dequeue_and_enqueue(k_loc_update, self._buffers["queue_loc"], self._buffers["queue_ptr_loc"])

        ### json log
        filename = 'l_mean_gl_v96.json'
        data = {'l_pos_mean': str(l_pos_gl_mean), 'l_neg_mean': str(l_neg_gl_mean), 'l_neg_var': str(l_neg_gl_var),
                'l_mix_gl': str(l_mix_gl),
                'l_pos_loc_mean': str(l_pos_loc_mean), 'l_neg_loc_mean': str(l_neg_loc_mean), 'l_neg_loc_var': str(l_neg_loc_var),
                'l_mix_loc': str(l_mix_loc),
                }
        with open(filename, 'a+') as file_obj:
            json.dump(data, file_obj)
            file_obj.write('\n')

        return losses


