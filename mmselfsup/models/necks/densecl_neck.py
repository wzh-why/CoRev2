# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.runner import BaseModule

from ..builder import NECKS


@NECKS.register_module()
class DenseCLNeck(BaseModule):
    """The non-linear neck of DenseCL.

    Single and dense neck in parallel: fc-relu-fc, conv-relu-conv.
    Borrowed from the authors' code: `<https://github.com/WXinlong/DenseCL`_.

    Args:
        in_channels (int): Number of input channels.
        hid_channels (int): Number of hidden channels.
        out_channels (int): Number of output channels.
        num_grid (int): The grid size of dense features. Defaults to None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 num_grid=None,
                 init_cfg=None):
        super(DenseCLNeck, self).__init__(init_cfg)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

        self.with_pool = True if num_grid is not None else False
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))
        self.mlp2 = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """Forward function of neck.

        Args:
            x (list[tensor]): feature map of backbone.
        """
        assert len(x) == 1
        x = x[0]

        avgpooled_x = self.avgpool(x)
        avgpooled_x = self.mlp(avgpooled_x.view(avgpooled_x.size(0), -1))   ## 这个部分是 全局分支映射头后的的输出

        print('before mlp2 x shape is', x.size())
        if self.with_pool:
            x = self.pool(x)  # sxs   ## 只是为了控制要将 feature map 切换为多少网格
        x = self.mlp2(x)  # sxs: bxdxsxs   ## 这个部分是局部密集映射头后的输出
        print('before mlp2 av_x shape is', avgpooled_x.size())
        print('after mlp2 x shape is',x.size())

        avgpooled_x2 = self.avgpool2(x)  # 1x1: bxdx1x1
        x = x.view(x.size(0), x.size(1), -1)  # bxdxs^2
        avgpooled_x2 = avgpooled_x2.view(avgpooled_x2.size(0), -1)  # bxd 这个部分是将网格状的特征向量进行全局池化后，加入到队列里进行更新队列
        return [avgpooled_x, x, avgpooled_x2]
