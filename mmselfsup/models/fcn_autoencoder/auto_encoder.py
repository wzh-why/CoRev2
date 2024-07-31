'''搭建 ae 的类'''
from torch import nn

from models.fcn_autoencoder.resnet import resnet50
from models.fcn_autoencoder.unet import unetUp


class fcn_resnet(nn.Module):
    def __init__(self,pretrained = False):
        super(fcn_resnet,self).__init__()
        self.encoder = resnet50(pretrained = pretrained)

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
            nn.ReLU(),
            nn.Conv2d(out_filters[0], out_filters[0], kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.con_final = nn.Conv2d(out_filters[0],3,kernel_size=1)


    def forward(self, inputs):

        [feat1, feat2, feat3, feat4, feat5] = self.encoder.forward(inputs)

        up4 = self.up_concat4(feat4, feat5)
        up3 = self.up_concat3(feat3, up4)
        up2 = self.up_concat2(feat2, up3)
        up1 = self.up_concat1(feat1, up2)

        if self.up_conv != None:
            up1 = self.up_conv(up1)  # up1 shape is (img_h,img_w,64)
        ## 再添加一个卷积，使得 图片通道数为3
        re_img = self.con_final(up1)

        return re_img

### test
import torch

x = torch.ones(1,3,224,224)
model = fcn_resnet(pretrained=False)
Y = model(x)
print(Y.shape)