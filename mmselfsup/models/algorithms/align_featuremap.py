import torch
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
import random
import torch.nn as nn

class UnNormalize:
    #restore from T.Normalize
    #反归一化 to show tensor --> image
    def __init__(self,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225)):
        self.mean=torch.tensor(mean).view((1,-1,1,1))
        self.std=torch.tensor(std).view((1,-1,1,1))
    def __call__(self,x):
        x=(x*self.std)+self.mean
        return torch.clip(x,0,None)

def align_featuremap_dist(feature_list,random_order_list):

    ### 该函数是 指 每张图片的组合的顺序都是不同的
    ## accorading to mode, to align the feature map, for k_features
    features = feature_list[0]### include B,C,H,W
    feature_c,features_h,features_w = features.size()[1:4]
    random_order_list_num = random_order_list.size(0)
    avepooling = nn.AdaptiveAvgPool2d((1,1))

    for i in range(random_order_list_num):
        feature = features[i,:,:,:]
        random_order = random_order_list[i]
        if random_order == 0:  ###4321
            feature_1 = feature[:,0:5,0:5]
            feature_2 = feature[:,0:5,5:7]
            feature_3 = feature[:,5:7, 0:5]
            feature_4 = feature[:,5:7, 5:7]
        elif random_order == 1:  ###4231
            feature_1 = feature[:,0:5,0:5]
            feature_2 = feature[:,0:5,5:7]
            feature_3 = feature[:,5:7, 0:5]
            feature_4 = feature[:,5:7, 5:7]
        elif random_order == 2:  ### 3412
            feature_1 = feature[:, 0:5, 0:2]
            feature_2 = feature[:, 0:5, 2:7]
            feature_3 = feature[:, 5:7, 5:7]
            feature_4 = feature[:, 5:7, 0:5]
        elif random_order == 3:  ###2143
            feature_1 = feature[:, 0:2, 0:5]
            feature_2 = feature[:, 0:2, 5:7]
            feature_3 = feature[:, 2:7, 0:5]
            feature_4 = feature[:, 2:7, 5:7]
        elif random_order == 4:  ### 1324
            feature_1 = feature[:, 0:5, 0:5]
            feature_2 = feature[:, 0:5, 5:7]
            feature_3 = feature[:, 5:7, 0:5]
            feature_4 = feature[:, 5:7, 5:7]
        elif random_order == 5:  ### 3142
            feature_1 = feature[:, 0:5, 0:2]
            feature_2 = feature[:, 0:5, 2:7]
            feature_3 = feature[:, 5:7, 0:2]
            feature_4 = feature[:, 5:7, 2:7]
        elif random_order == 6:  ### 4132
            feature_1 = feature[:, 0:2, 0:2]
            feature_2 = feature[:, 0:5, 2:7]
            feature_3 = feature[:, 2:7, 0:2]
            feature_4 = feature[:, 5:7, 2:7]
        elif random_order == 7:  ### 1432
            feature_1 = feature[:, 0:5, 0:5]
            feature_2 = feature[:, 0:2, 5:7]
            feature_3 = feature[:, 5:7, 0:5]
            feature_4 = feature[:, 2:7, 5:7]

        ### add avapooling to feature_i to ensure the same dimension
        feature_1 = avepooling(feature_1)
        feature_2 = avepooling(feature_2)
        feature_3 = avepooling(feature_3)
        feature_4 = avepooling(feature_4)
        if i == 0:
            feature_1_tensor = feature_1
            feature_2_tensor = feature_2
            feature_3_tensor = feature_3
            feature_4_tensor = feature_4
        else:
            feature_1_tensor = torch.cat((feature_1_tensor,feature_1),dim=0)
            feature_2_tensor = torch.cat((feature_2_tensor, feature_2), dim=0)
            feature_3_tensor = torch.cat((feature_3_tensor, feature_3), dim=0)
            feature_4_tensor = torch.cat((feature_4_tensor, feature_4), dim=0)

    feature_1_final = feature_1_tensor.view(-1,feature_c,1,1)
    feature_2_final = feature_2_tensor.view(-1, feature_c, 1, 1)
    feature_3_final = feature_3_tensor.view(-1, feature_c, 1, 1)
    feature_4_final = feature_4_tensor.view(-1, feature_c, 1, 1)

    return feature_1_final,feature_2_final,feature_3_final,feature_4_final

def align_featuremap_dist_fpn(feature_list,random_order_list):

    ### 该函数是 指 每张图片的组合的顺序都是不同的
    ## accorading to mode, to align the feature map, for k_features
    features = feature_list[0]### include B,C,H,W
    feature_c,features_h,features_w = features.size()[1:4]
    random_order_list_num = random_order_list.size(0)
    avepooling = nn.AdaptiveAvgPool2d((1,1))

    for i in range(random_order_list_num):
        feature = features[i,:,:,:]
        random_order = random_order_list[i]
        if random_order == 0:  ###4321
            feature_1 = feature[:,0:10,0:10]
            feature_2 = feature[:,0:10,10:14]
            feature_3 = feature[:,10:14, 0:10]
            feature_4 = feature[:,10:14, 10:14]
        elif random_order == 1:  ###4231
            feature_1 = feature[:,0:10,0:10]
            feature_2 = feature[:,0:10,10:14]
            feature_3 = feature[:,10:14, 0:10]
            feature_4 = feature[:,10:14, 10:14]
        elif random_order == 2:  ### 3412
            feature_1 = feature[:, 0:10, 0:4]
            feature_2 = feature[:, 0:10, 4:14]
            feature_3 = feature[:, 10:14, 10:14]
            feature_4 = feature[:, 10:14, 0:10]
        elif random_order == 3:  ###2143
            feature_1 = feature[:, 0:4, 0:10]
            feature_2 = feature[:, 0:4, 10:14]
            feature_3 = feature[:, 4:14, 0:10]
            feature_4 = feature[:, 4:14, 10:14]
        elif random_order == 4:  ### 1324
            feature_1 = feature[:, 0:10, 0:10]
            feature_2 = feature[:, 0:10, 10:14]
            feature_3 = feature[:, 10:14, 0:10]
            feature_4 = feature[:, 10:14, 10:14]
        elif random_order == 5:  ### 3142
            feature_1 = feature[:, 0:10, 0:4]
            feature_2 = feature[:, 0:10, 4:14]
            feature_3 = feature[:, 10:14, 0:4]
            feature_4 = feature[:, 10:14, 4:14]
        elif random_order == 6:  ### 4132
            feature_1 = feature[:, 0:4, 0:4]
            feature_2 = feature[:, 0:10, 4:14]
            feature_3 = feature[:, 4:14, 0:4]
            feature_4 = feature[:, 10:14, 4:14]
        elif random_order == 7:  ### 1432
            feature_1 = feature[:, 0:10, 0:10]
            feature_2 = feature[:, 0:4, 10:14]
            feature_3 = feature[:, 10:14, 0:10]
            feature_4 = feature[:, 4:14, 10:14]

        ### add avapooling to feature_i to ensure the same dimension
        feature_1 = avepooling(feature_1)
        feature_2 = avepooling(feature_2)
        feature_3 = avepooling(feature_3)
        feature_4 = avepooling(feature_4)
        if i == 0:
            feature_1_tensor = feature_1
            feature_2_tensor = feature_2
            feature_3_tensor = feature_3
            feature_4_tensor = feature_4
        else:
            feature_1_tensor = torch.cat((feature_1_tensor,feature_1),dim=0)
            feature_2_tensor = torch.cat((feature_2_tensor, feature_2), dim=0)
            feature_3_tensor = torch.cat((feature_3_tensor, feature_3), dim=0)
            feature_4_tensor = torch.cat((feature_4_tensor, feature_4), dim=0)

    feature_1_final = feature_1_tensor.view(-1,feature_c,1,1)
    feature_2_final = feature_2_tensor.view(-1, feature_c, 1, 1)
    feature_3_final = feature_3_tensor.view(-1, feature_c, 1, 1)
    feature_4_final = feature_4_tensor.view(-1, feature_c, 1, 1)

    return feature_1_final,feature_2_final,feature_3_final,feature_4_final

# ### test
#
# input = torch.ones(16,2048,7,7)
# random_list = torch.tensor([1,2,3,4,5,6,7,8,2,3,4,5,6,7,8,0,])
# align_featuremap(feature_list=[input],random_order_list=random_list)

#
###  src alignment
def align_featuremap(feature_list,random_order):

    ### 这个算法指的是 每个 batch_size 的图像都经过了一样的 组合顺序
    ## accorading to mode, to align the feature map, for k_features
    feature = feature_list[0]  ### include B,C,H,W 将 feature 从list中取出

    if random_order == 0:  ###4321
        feature_4 = feature[:,:,0:5,0:5]
        feature_3 = feature[:,:,0:5,5:7]
        feature_2 = feature[:,:,5:7, 0:5]
        feature_1 = feature[:,:,5:7, 5:7]
    elif random_order == 1:  ###4231
        feature_4 = feature[:,:,0:5,0:5]
        feature_2 = feature[:,:,0:5,5:7]
        feature_3 = feature[:,:,5:7, 0:5]
        feature_1 = feature[:,:,5:7, 5:7]
    elif random_order == 2:  ### 3412
        feature_3 = feature[:, :, 0:5, 0:2]
        feature_4 = feature[:, :, 0:5, 2:7]
        feature_1 = feature[:, :, 5:7, 5:7]
        feature_2 = feature[:, :, 5:7, 0:5]
    elif random_order == 3:  ###2143
        feature_2 = feature[:, :, 0:2, 0:5]
        feature_1 = feature[:, :, 0:2, 5:7]
        feature_4 = feature[:, :, 2:7, 0:5]
        feature_3 = feature[:, :, 2:7, 5:7]
    elif random_order == 4:  ### 1324
        feature_1 = feature[:, :, 0:5, 0:5]
        feature_3 = feature[:, :, 0:5, 5:7]
        feature_2 = feature[:, :, 5:7, 0:5]
        feature_4 = feature[:, :, 5:7, 5:7]
    elif random_order == 5:  ### 3142
        feature_3 = feature[:, :, 0:5, 0:2]
        feature_1 = feature[:, :, 0:5, 2:7]
        feature_4 = feature[:, :, 5:7, 0:2]
        feature_2 = feature[:, :, 5:7, 2:7]
    elif random_order == 6:  ### 4132
        feature_4 = feature[:, :, 0:2, 0:2]
        feature_1 = feature[:, :, 0:5, 2:7]
        feature_3 = feature[:, :, 2:7, 0:2]
        feature_2 = feature[:, :, 5:7, 2:7]
    elif random_order == 7:  ### 1432
        feature_1 = feature[:, :, 0:5, 0:5]
        feature_4 = feature[:, :, 0:2, 5:7]
        feature_3 = feature[:, :, 5:7, 0:5]
        feature_2 = feature[:, :, 2:7, 5:7]

    return feature_1,feature_2,feature_3,feature_4


def random_montage_4321(img):

    ### tensor B,C,H,W
    ### input: Tensor (3,224,224)
    img_patch_1 = img[:,:,0:64, 0:64]  # 3 channel need to copy
    img_patch_2 = img[:,:,0:64, 64:224]
    img_patch_3 = img[:,:,64:224, 0:64]
    img_patch_4 = img[:,:,64:224, 64:224]

    pipeline_montage = transforms.Compose([
            ## keep the transforms with mocov2
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomRotation(degrees=(-15, 18)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    img_patch_1 = pipeline_montage(img_patch_1)
    img_patch_2 = pipeline_montage(img_patch_2)
    img_patch_3 = pipeline_montage(img_patch_3)
    img_patch_4 = pipeline_montage(img_patch_4)

    img2 = torch.zeros_like(img)

    img2[:,:,0:160, 0:160] = img_patch_4
    img2[:,:,0:160, 160:224] = img_patch_3
    img2[:,:,160:224, 0:160] = img_patch_2
    img2[:,:,160:224, 160:224] = img_patch_1

    # # just for show
    # img2 = img2[1,:,:,:]
    # print('img1 shape is',img2.size())
    # img2 = img2.cpu()
    # img2_unnor = UnNormalize()(img2)
    # img2_unnor = torch.squeeze(img2_unnor,dim=0)
    # img2_PIL = transforms.ToPILImage()(img2_unnor)
    # img2_PIL.show()
    # kkk

    return img2

def random_montage_4231(img):

    ### tensor B,C,H,W
    ### input: Tensor (3,224,224)
    img_patch_1 = img[:, :, 0:64, 0:64]  # 3 channel need to copy
    img_patch_2 = img[:, :, 0:64, 64:224].permute([0,1,3,2])
    img_patch_3 = img[:, :, 64:224, 0:64].permute([0,1,3,2])
    img_patch_4 = img[:, :, 64:224, 64:224]

    pipeline_montage = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    img_patch_1 = pipeline_montage(img_patch_1)
    img_patch_2 = pipeline_montage(img_patch_2)
    img_patch_3 = pipeline_montage(img_patch_3)
    img_patch_4 = pipeline_montage(img_patch_4)

    img2 = torch.zeros_like(img)

    #  do not need shuffle
    img2[:,:,0:160, 0:160] = img_patch_4
    img2[:,:,0:160, 160:224] = img_patch_2
    img2[:,:,160:224, 0:160] = img_patch_3
    img2[:,:,160:224, 160:224] = img_patch_1

    # # just for show
    # img2 = img2[1,:,:,:]
    # print('img1 shape is',img2.size())
    # img2 = img2.cpu()
    # img2_unnor = UnNormalize()(img2)
    # img2_unnor = torch.squeeze(img2_unnor,dim=0)
    # img2_PIL = transforms.ToPILImage()(img2_unnor)
    # img2_PIL.show()
    # kkk
    return img2

def random_montage_2143(img):
    ### tensor B,C,H,W
    ### input: Tensor (3,224,224)
    img_patch_1 = img[:, :, 0:64, 0:64]  # 3 channel need to copy
    img_patch_2 = img[:, :, 0:64, 64:224]
    img_patch_3 = img[:, :, 64:224, 0:64]
    img_patch_4 = img[:, :, 64:224, 64:224]

    pipeline_montage = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    img_patch_1 = pipeline_montage(img_patch_1)
    img_patch_2 = pipeline_montage(img_patch_2)
    img_patch_3 = pipeline_montage(img_patch_3)
    img_patch_4 = pipeline_montage(img_patch_4)

    img2 = torch.zeros_like(img)
    #  do not need shuffle
    img2[:,:,64:224, 0:160] = img_patch_4
    img2[:,:,64:224, 160:224] = img_patch_3
    img2[:,:,0:64, 0:160] = img_patch_2
    img2[:,:,0:64, 160:224] = img_patch_1

    # # just for show
    # img2 = img2[1,:,:,:]
    # print('img1 shape is',img2.size())
    # img2 = img2.cpu()
    # img2_unnor = UnNormalize()(img2)
    # img2_unnor = torch.squeeze(img2_unnor,dim=0)
    # img2_PIL = transforms.ToPILImage()(img2_unnor)
    # img2_PIL.show()
    # kkk
    return img2

def random_montage_3412(img):

    ### tensor B,C,H,W
    ### input: Tensor (3,224,224)
    img_patch_1 = img[:, :, 0:64, 0:64]  # 3 channel need to copy
    img_patch_2 = img[:, :, 0:64, 64:224]
    img_patch_3 = img[:, :, 64:224, 0:64]
    img_patch_4 = img[:, :, 64:224, 64:224]

    pipeline_montage = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    img_patch_1 = pipeline_montage(img_patch_1)
    img_patch_2 = pipeline_montage(img_patch_2)
    img_patch_3 = pipeline_montage(img_patch_3)
    img_patch_4 = pipeline_montage(img_patch_4)

    img2 = torch.zeros_like(img)

    #  do not need shuffle
    img2[:,:,0:160, 64:224] = img_patch_4
    img2[:,:,0:160, 0:64] = img_patch_3
    img2[:,:,160:224,64:224] = img_patch_2
    img2[:,:,160:224, 0:64] = img_patch_1

    # # just for show
    # img2 = img2[1,:,:,:]
    # print('img1 shape is',img2.size())
    # img2 = img2.cpu()
    # img2_unnor = UnNormalize()(img2)
    # img2_unnor = torch.squeeze(img2_unnor,dim=0)
    # img2_PIL = transforms.ToPILImage()(img2_unnor)
    # img2_PIL.show()
    # kkk
    return img2

def random_montage_size_1324(img):

    img_patch_1 = img[:, :, 0:64, 0:64]  # 3 channel need to copy
    img_patch_2 = img[:, :, 0:64, 64:224]
    img_patch_3 = img[:, :, 64:224, 0:64]
    img_patch_4 = img[:, :, 64:224, 64:224]

    pipeline_montage_1 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.Resize(size=(160,160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    pipeline_montage_23 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    pipeline_montage_4 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.Resize(size=(64,64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    img_patch_1 = pipeline_montage_1(img_patch_1)
    img_patch_2 = pipeline_montage_23(img_patch_2)
    img_patch_3 = pipeline_montage_23(img_patch_3)
    img_patch_4 = pipeline_montage_4(img_patch_4)

    img2 = torch.zeros_like(img)
    #  do not need shuffle
    img2[:,:,0:160, 0:160] = img_patch_1
    img2[:,:,0:160, 160:224] = img_patch_3
    img2[:,:,160:224,0:160] = img_patch_2
    img2[:,:,160:224, 160:224] = img_patch_4

    # # just for show
    # img2 = img2[1,:,:,:]
    # print('img1 shape is',img2.size())
    # img2 = img2.cpu()
    # img2_unnor = UnNormalize()(img2)
    # img2_unnor = torch.squeeze(img2_unnor,dim=0)
    # img2_PIL = transforms.ToPILImage()(img2_unnor)
    # img2_PIL.show()
    # kkk

    return img2

def random_montage_size_3142(img):

    img_patch_1 = img[:, :, 0:64, 0:64]  # 3 channel need to copy
    img_patch_2 = img[:, :, 0:64, 64:224]
    img_patch_3 = img[:, :, 64:224, 0:64]
    img_patch_4 = img[:, :, 64:224, 64:224]

    pipeline_montage_1 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.Resize(size=(160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    pipeline_montage_23 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    pipeline_montage_4 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    img_patch_1 = pipeline_montage_1(img_patch_1)
    img_patch_2 = pipeline_montage_23(img_patch_2)
    img_patch_3 = pipeline_montage_23(img_patch_3)
    img_patch_4 = pipeline_montage_4(img_patch_4)

    img2 = torch.zeros_like(img)

    #  do not need shuffle
    img2[:,:,0:160, 0:64] = img_patch_3
    img2[:,:,0:160, 64:224] = img_patch_1
    img2[:,:,160:224,0:64] = img_patch_4
    img2[:,:,160:224, 64:224] = img_patch_2

    # # just for show
    # img2 = img2[1,:,:,:]
    # print('img1 shape is',img2.size())
    # img2 = img2.cpu()
    # img2_unnor = UnNormalize()(img2)
    # img2_unnor = torch.squeeze(img2_unnor,dim=0)
    # img2_PIL = transforms.ToPILImage()(img2_unnor)
    # img2_PIL.show()
    # kkk

    return img2


def random_montage_size_4132(img):
    img_patch_1 = img[:, :, 0:64, 0:64]  # 3 channel need to copy
    img_patch_2 = img[:, :, 0:64, 64:224]
    img_patch_3 = img[:, :, 64:224, 0:64]
    img_patch_4 = img[:, :, 64:224, 64:224]

    pipeline_montage_1 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.Resize(size=(160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    pipeline_montage_23 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    pipeline_montage_4 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    img_patch_1 = pipeline_montage_1(img_patch_1)
    img_patch_2 = pipeline_montage_23(img_patch_2)
    img_patch_3 = pipeline_montage_23(img_patch_3)
    img_patch_4 = pipeline_montage_4(img_patch_4)

    img2 = torch.zeros_like(img)

    #  do not need shuffle
    img2[:,:,0:64, 0:64] = img_patch_4
    img2[:,:,0:160, 64:224] = img_patch_1
    img2[:,:,64:224,0:64] = img_patch_3
    img2[:,:,160:224, 64:224] = img_patch_2

    # # just for show
    # img2 = img2[1, :, :, :]
    # print('img1 shape is', img2.size())
    # img2 = img2.cpu()
    # img2_unnor = UnNormalize()(img2)
    # img2_unnor = torch.squeeze(img2_unnor, dim=0)
    # img2_PIL = transforms.ToPILImage()(img2_unnor)
    # img2_PIL.show()
    # kkk
    return img2


def random_montage_size_1432(img):

    img_patch_1 = img[:, :, 0:64, 0:64]  # 3 channel need to copy
    img_patch_2 = img[:, :, 0:64, 64:224].permute([0, 1, 3, 2])
    img_patch_3 = img[:, :, 64:224, 0:64].permute([0, 1, 3, 2])
    img_patch_4 = img[:, :, 64:224, 64:224]

    pipeline_montage_1 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.Resize(size=(160, 160)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    pipeline_montage_23 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    pipeline_montage_4 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])

    img_patch_1 = pipeline_montage_1(img_patch_1)
    img_patch_2 = pipeline_montage_23(img_patch_2)
    img_patch_3 = pipeline_montage_23(img_patch_3)
    img_patch_4 = pipeline_montage_4(img_patch_4)

    img2 = torch.zeros_like(img)

    #  do not need shuffle
    img2[:,:,0:160, 0:160] = img_patch_1
    img2[:,:,0:64, 160:224] = img_patch_4
    img2[:,:,160:224,0:160] = img_patch_3
    img2[:,:,64:224, 160:224] = img_patch_2

    # # just for show
    # img2 = img2[1, :, :, :]
    # print('img1 shape is', img2.size())
    # img2 = img2.cpu()
    # img2_unnor = UnNormalize()(img2)
    # img2_unnor = torch.squeeze(img2_unnor, dim=0)
    # img2_PIL = transforms.ToPILImage()(img2_unnor)
    # img2_PIL.show()
    # kkk

    return img2


def random_choice(img):

    ## input img: tensor [B,C,H,W]

    random_order = random.randint(0, 7)
    ##
    if random_order == 0:  ###4321
        img = random_montage_4321(img)
        order = 0
    elif random_order == 1:  ###4231
        img = random_montage_4231(img)
        order = 1
        # label = torch.LongTensor([3,1,2,0])
    elif random_order == 2:  ### 3412
        img = random_montage_3412(img)
        # label = torch.LongTensor([2, 3, 0, 1])
        order = 2
    elif random_order == 3:  ###2143
        img = random_montage_2143(img)
        # label = torch.LongTensor([1, 0, 3, 2])
        order = 3
    elif random_order == 4:  ### 1324
        img = random_montage_size_1324(img)
        order = 4
    elif random_order == 5:  ### 3142
        img = random_montage_size_3142(img)
        order = 5
    elif random_order == 6:  ### 4132
        img = random_montage_size_4132(img)
        order = 6
    elif random_order == 7:  ### 1432
        img = random_montage_size_1432(img)
        order = 7

    return img,order
