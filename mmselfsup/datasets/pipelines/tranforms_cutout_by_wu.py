'''
it add cutout(a window)  at the center of image and return the position of window
'''

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import random
import time

class UnNormalize:
    #restore from T.Normalize
    #反归一化 to show tensor --> image
    def __init__(self,mean=(0.485, 0.456, 0.406),std= (0.229, 0.224, 0.225)):
        self.mean=torch.tensor(mean).view((1,-1,1,1))
        self.std=torch.tensor(std).view((1,-1,1,1))
    def __call__(self,x):
        x=(x*self.std)+self.mean
        return torch.clip(x,0,None)

def cutout_center(img):

    ## give the PIL image
    # obtain the h and w of image
    img_h,img_w = img.size()[1:3]
    img3 = img.clone()
    window_h = int(0.25*img_h)
    window_w = int(0.25*img_w)
    img_center_h = int(0.5 * img_h)
    img_center_w = int(0.5*img_w)
    window_pos = [img_center_h-int(0.5*window_h),img_center_w-int(0.5*window_w),img_center_h+int(0.5*window_h),img_center_w+int(0.5*window_w)]    # [(x1,y1),(x2,y2)]
    img3[:,window_pos[0]:window_pos[2],window_pos[1]:window_pos[3]] = 0

    return img3

# def transform_center(img):
#
#     img_h,img_w = img.size()[1:3]
#     window_h = int(0.25*img_h)
#     window_w = int(0.25*img_w)
#     img_center_h = int(0.5 * img_h)
#     img_center_w = int(0.5 * img_w)
#
#     img3 = img.clone()  ### copy
#
#     ## window postion can be random set
#
#     window_pos = [img_center_h-int(0.5*window_h),img_center_w-int(0.5*window_w),img_center_h+int(0.5*window_h),img_center_w+int(0.5*window_w)]    # [(x1,y1),(x2,y2)]
#
#     img_center_transforms = transforms.Compose([
#         # transforms.RandomChoice([
#         #     transforms.RandomHorizontalFlip(p=0.5),
#         #     transforms.ColorJitter(brightness=0.4,
#         #         contrast=0.4,
#         #         saturation=0.4,
#         #         hue=0.1),
#         #     transforms.RandomRotation(degrees=(-15,18))],
#         #
#         # ),
#         ## keep the transforms with mocov2
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.ColorJitter(brightness=0.4,
#                                contrast=0.4,
#                                saturation=0.4,
#                                hue=0.1),
#         transforms.RandomGrayscale(p=0.2),
#         transforms.RandomRotation(degrees=(-15, 18)),
#         transforms.GaussianBlur(kernel_size=3,sigma=(0.1,2.0)),
#         # transforms.ToPILImage(),
#     ])
#
#     img_center = img[:,window_pos[0]:window_pos[2],window_pos[1]:window_pos[3]]  ## obtain the area [3,64,64]
#
#     img_center_tran = img_center_transforms(img_center)
#
#     img3[:,window_pos[0]:window_pos[2], window_pos[1]:window_pos[3]] = img_center_tran
#
#
#     ## just for show
#     # img1_unnor = UnNormalize()(img)
#     # img1_unnor = torch.squeeze(img1_unnor,dim=0)
#     # img1_PIL = transforms.ToPILImage()(img1_unnor)
#     # img1_PIL.show()
#     # img1 = img1_PIL.save('img1_PIL.jpg')
#     #
#     # img3_unnor = UnNormalize()(img3)
#     # img3_unnor = torch.squeeze(img3_unnor, dim=0)
#     # img3_PIL = transforms.ToPILImage()(img3_unnor)
#     # img3_PIL.show()
#     # img3 = img3_PIL.save('img3_PIL.jpg')
#
#     return img3

def transform_center(img):

    img_h, img_w = img.size[:2]
    window_h = int(0.25 * img_h)
    window_w = int(0.25 * img_w)
    img_center_h = int(0.5 * img_h)
    img_center_w = int(0.5 * img_w)

    ## window postion can be random set

    window_pos = [img_center_h - int(0.5 * window_h), img_center_w - int(0.5 * window_w),
                  img_center_h + int(0.5 * window_h), img_center_w + int(0.5 * window_w)]  # [(x1,y1),(x2,y2)]

    # img to ndaary
    img = np.array(img)

    img_center_transforms = transforms.Compose([

        ## keep the transforms with mocov2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img_center = img[window_pos[0]:window_pos[2], window_pos[1]:window_pos[3]]

    ## first transform into
    img_center = Image.fromarray(img_center, mode='RGB')

    img1_c = transforms.ToPILImage()(img_center_transforms(img_center))

    img[window_pos[0]:window_pos[2], window_pos[1]:window_pos[3]] = img1_c

    # from ndary to img
    img3 = Image.fromarray(img.astype('uint8')).convert('RGB')

    img3 = totensor(img3)

    return img3

def cutout_subfigure_center(img):

    ## give the PIL image
    # obtain the h and w of image
    img_h,img_w = img.size()[1:3]
    img3 = img.clone()

    # first subfigure
    window_h_1 = 32
    window_w_1 = 32
    img_center_h_1 = 32
    img_center_w_1 = 32

    # second  subfigure
    window_h_2 = 32
    window_w_2 = 64
    img_center_h_2 = 32
    img_center_w_2 = 144

    # third  subfigure
    window_h_3 = 64
    window_w_3 = 32
    img_center_h_3 = 144
    img_center_w_3 = 32

    # fourth  subfigure
    window_h_4 = 64
    window_w_4 = 64
    img_center_h_4 = 144
    img_center_w_4 = 144

    ## window postion can be random set
    window_pos_1 = [img_center_h_1-int(0.5*window_h_1),img_center_w_1-int(0.5*window_w_1),img_center_h_1+int(0.5*window_h_1),img_center_w_1+int(0.5*window_w_1)]    # [(x1,y1),(x2,y2)]
    window_pos_2 = [img_center_h_2 - int(0.5 * window_h_2), img_center_w_2 - int(0.5 * window_w_2),
                    img_center_h_2 + int(0.5 * window_h_2), img_center_w_2 + int(0.5 * window_w_2)]

    window_pos_3 = [img_center_h_3 - int(0.5 * window_h_3), img_center_w_3 - int(0.5 * window_w_3),
                    img_center_h_3 + int(0.5 * window_h_3), img_center_w_3 + int(0.5 * window_w_3)]

    window_pos_4 = [img_center_h_4 - int(0.5 * window_h_4), img_center_w_4 - int(0.5 * window_w_4),
                    img_center_h_4 + int(0.5 * window_h_4), img_center_w_4 + int(0.5 * window_w_4)]

    img3[:,window_pos_1[0]:window_pos_1[2],window_pos_1[1]:window_pos_1[3]] = 0
    img3[:,window_pos_2[0]:window_pos_2[2], window_pos_2[1]:window_pos_2[3]] = 0
    img3[:,window_pos_3[0]:window_pos_3[2], window_pos_3[1]:window_pos_3[3]] = 0
    img3[:,window_pos_4[0]:window_pos_4[2], window_pos_4[1]:window_pos_4[3]] = 0

    # ## just for visualize
    # img3_unnor = UnNormalize()(img3)
    # img3_unnor = torch.squeeze(img3_unnor, dim=0)
    # img3_PIL = transforms.ToPILImage()(img3_unnor)
    # img3_PIL.show()
    # img3 = img3_PIL.save('img3_PIL.jpg')

    return img3


def random_subfigure_cutout(img):
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy  H,W,C
    img_patch_2 = img[0:64, 64:224, :]
    img_patch_3 = img[64:224, 0:64, :]
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

    # define some tran
    pipeline_montage_1 = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomErasing(p=1,scale=(0.1,0.2)),
        transforms.RandomGrayscale(p=0.2),
    ])

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img2_1 = transforms.ToPILImage()(pipeline_montage_1(img_patch_1))
    img2_2 = transforms.ToPILImage()(pipeline_montage_1(img_patch_2))  # tran
    img2_3 = transforms.ToPILImage()(pipeline_montage_1(img_patch_3))
    img2_4 = transforms.ToPILImage()(pipeline_montage_1(img_patch_4))

    ##  do not need shuffle
    img2[0:64, 0:64] = img2_1
    img2[0:64, 64:224] = img2_2
    img2[64:224, 0:64] = img2_3
    img2[64:224, 64:224] = img2_4

    ### create mask
    img_cut_area = img2-img
    mask = img_cut_area
    mask[mask!=0]=255   ### denotes cutout area

    # # ## add to show
    # mask_inv = cv2.bitwise_not(mask)
    #
    # image_spilt = np.multiply(img, mask)
    # # img1 = Image.fromarray(img1)
    # img2 = Image.fromarray(image_spilt)
    # img2.show()
    # # img1 = totensor(img1)

    img4 = totensor(img2)
    return img4,mask

def montage(img):

    ### input: Tensor (3,224,224)
    img_patch_1 = img[:,0:64, 0:64]  # 3 channel need to copy
    img_patch_2 = img[:,0:64, 64:224]
    img_patch_3 = img[:,64:224, 0:64]
    img_patch_4 = img[:,64:224, 64:224]

    pipeline_montage = transforms.Compose([
            ## keep the transforms with mocov2
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.ColorJitter(brightness=0.4,
            #                        contrast=0.4,
            #                        saturation=0.4,
            #                        hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomRotation(degrees=(-15, 18)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    img2_patch_1 = pipeline_montage(img_patch_1)
    img2_patch_2 = pipeline_montage(img_patch_2)
    img2_patch_3 = pipeline_montage(img_patch_3)
    img2_patch_4 = pipeline_montage(img_patch_4)

    img2 = torch.zeros_like(img)
    img2[:, 0:64, 0:64] = img2_patch_1  # 3 channel need to copy
    img2[:, 0:64, 64:224] = img2_patch_2
    img2[:, 64:224, 0:64] = img2_patch_3
    img2[:, 64:224, 64:224] = img2_patch_4

    return img2

def montage_PIL(img):

    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy  H,W,C
    img_patch_2 = img[0:64, 64:224, :]
    img_patch_3 = img[64:224, 0:64, :]
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

    # define some tran
    pipeline_montage_1 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomResizedCrop(size=(64,64),scale=(0.5, 1.)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    pipeline_montage_2 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomResizedCrop(size=(64, 160), scale=(0.2, 0.6)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    pipeline_montage_3 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomResizedCrop(size=(160,64), scale=(0.2, 0.6)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])

    pipeline_montage_4 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomResizedCrop(size=(160,160), scale=(0.1, 0.5)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img2_1 = transforms.ToPILImage()(pipeline_montage_1(img_patch_1))
    img2_2 = transforms.ToPILImage()(pipeline_montage_2(img_patch_2))  # tran
    img2_3 = transforms.ToPILImage()(pipeline_montage_3(img_patch_3))
    img2_4 = transforms.ToPILImage()(pipeline_montage_4(img_patch_4))
    # img2 = torch.zeros(size=(224, 224, 3))

    ##  do not need shuffle
    img2[0:64, 0:64] = img2_1
    img2[0:64, 64:224] = img2_2
    img2[64:224, 0:64] = img2_3
    img2[64:224, 64:224] = img2_4

    # ## add to show
    # img1 = Image.fromarray(img1)
    # img2 = Image.fromarray(img2)
    # img1.show()
    # img2.show()

    #
    # # img1 = totensor(img1)
    img2 = totensor(img2)
    #
    return img2


def v5_global_trans(img):

    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)

    # define some tran
    pipeline_global_v5 = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomResizedCrop(size=(224,224),scale=(0.8, 1.)),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    img = Image.fromarray(img, mode='RGB')
    img2 = transforms.ToPILImage()(pipeline_global_v5(img))
    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img2)
    return img2

def v11_loc_trans(img):

    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy
    img_patch_2 = img[0:64, 64:224, :]
    img_patch_3 = img[64:224, 0:64, :]
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

    pipeline_montage = transforms.Compose([
        ## keep the transforms with mocov2
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomRotation(degrees=(-15, 18)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img_patch_1 = transforms.ToPILImage()(pipeline_montage(img_patch_1))
    img_patch_2 = transforms.ToPILImage()(pipeline_montage(img_patch_2))  # tran
    img_patch_3 = transforms.ToPILImage()(pipeline_montage(img_patch_3))
    img_patch_4 = transforms.ToPILImage()(pipeline_montage(img_patch_4))

    #  do not need shuffle
    img2[0:64, 0:64] = img_patch_1
    img2[0:64, 64:224] = img_patch_2
    img2[64:224, 0:64] = img_patch_3
    img2[64:224, 64:224] = img_patch_4

    # ### just to show
    # img2 = Image.fromarray(img2)
    # img2.show()

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    img2 = totensor(img2)

    return img2


def dataset_trans(img):

    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)

    # define some tran
    pipeline_global_v5 = transforms.Compose([
        ## keep the transforms with mocov2
        # transforms.RandomResizedCrop(size=(224,224),scale=(0.8, 1.)),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.RandomGrayscale(p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        # transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    img = Image.fromarray(img, mode='RGB')
    img2 = transforms.ToPILImage()(pipeline_global_v5(img))
    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img2)
    return img2

def random_montage_4321(img):

    ### PIL image --> np (H,W,C)

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy
    img_patch_2 = img[0:64, 64:224, :]
    img_patch_3 = img[64:224, 0:64, :]
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

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
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img_patch_1 = transforms.ToPILImage()(pipeline_montage(img_patch_1))
    img_patch_2 = transforms.ToPILImage()(pipeline_montage(img_patch_2))  # tran
    img_patch_3 = transforms.ToPILImage()(pipeline_montage(img_patch_3))
    img_patch_4 = transforms.ToPILImage()(pipeline_montage(img_patch_4))

    #  do not need shuffle
    img2[0:160, 0:160] = img_patch_4
    img2[0:160, 160:224] = img_patch_3
    img2[160:224, 0:160] = img_patch_2
    img2[160:224, 160:224] = img_patch_1

    # ### just to show
    # img2 = Image.fromarray(img2)
    # img2.show()

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    img2 = totensor(img2)

    return img2

def random_montage_4231(img):

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy
    img_patch_2 = img[0:64, 64:224, :].transpose(1,0,2)
    img_patch_3 = img[64:224, 0:64, :].transpose(1,0,2)
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

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
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img_patch_1 = transforms.ToPILImage()(pipeline_montage(img_patch_1))
    img_patch_2 = transforms.ToPILImage()(pipeline_montage(img_patch_2))  # tran
    img_patch_3 = transforms.ToPILImage()(pipeline_montage(img_patch_3))
    img_patch_4 = transforms.ToPILImage()(pipeline_montage(img_patch_4))

    #  do not need shuffle
    img2[0:160, 0:160] = img_patch_4
    img2[0:160, 160:224] = img_patch_2
    img2[160:224, 0:160] = img_patch_3
    img2[160:224, 160:224] = img_patch_1

    # ### just to show
    # img2 = Image.fromarray(img2)
    # img2.show()

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img2)

    return img2


def random_montage_2143(img):

    ### PIL image --> np (H,W,C)

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy
    img_patch_2 = img[0:64, 64:224, :]
    img_patch_3 = img[64:224, 0:64, :]
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

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

        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img_patch_1 = transforms.ToPILImage()(pipeline_montage(img_patch_1))
    img_patch_2 = transforms.ToPILImage()(pipeline_montage(img_patch_2))  # tran
    img_patch_3 = transforms.ToPILImage()(pipeline_montage(img_patch_3))
    img_patch_4 = transforms.ToPILImage()(pipeline_montage(img_patch_4))

    #  do not need shuffle
    img2[64:224, 0:160] = img_patch_4
    img2[64:224, 160:224] = img_patch_3
    img2[0:64, 0:160] = img_patch_2
    img2[0:64, 160:224] = img_patch_1

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img2)

    # img2 = Image.fromarray(img2)
    # img2.show()

    return img2

def random_montage_3412(img):

    ### PIL image --> np (H,W,C)

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy
    img_patch_2 = img[0:64, 64:224, :]
    img_patch_3 = img[64:224, 0:64, :]
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

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
        transforms.ToTensor(),
        # transforms.ToPILImage(),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img_patch_1 = transforms.ToPILImage()(pipeline_montage(img_patch_1))
    img_patch_2 = transforms.ToPILImage()(pipeline_montage(img_patch_2))  # tran
    img_patch_3 = transforms.ToPILImage()(pipeline_montage(img_patch_3))
    img_patch_4 = transforms.ToPILImage()(pipeline_montage(img_patch_4))

    #  do not need shuffle
    img2[0:160, 64:224] = img_patch_4
    img2[0:160, 0:64] = img_patch_3
    img2[160:224,64:224] = img_patch_2
    img2[160:224, 0:64] = img_patch_1

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img2)

    # img2 = Image.fromarray(img2)
    # img2.show()

    return img2

def random_montage_size_1324(img):

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy
    img_patch_2 = img[0:64, 64:224, :]
    img_patch_3 = img[64:224, 0:64, :]
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

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
        transforms.ToTensor(),
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
        transforms.ToTensor(),
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
        transforms.ToTensor(),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img_patch_1 = transforms.ToPILImage()(pipeline_montage_1(img_patch_1))
    img_patch_2 = transforms.ToPILImage()(pipeline_montage_23(img_patch_2))  # tran
    img_patch_3 = transforms.ToPILImage()(pipeline_montage_23(img_patch_3))
    img_patch_4 = transforms.ToPILImage()(pipeline_montage_4(img_patch_4))

    #  do not need shuffle
    img2[0:160, 0:160] = img_patch_1
    img2[0:160, 160:224] = img_patch_3
    img2[160:224,0:160] = img_patch_2
    img2[160:224, 160:224] = img_patch_4

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img2)

    # img2 = Image.fromarray(img2)
    # img2.show()

    return img2

def random_montage_size_3142(img):

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy
    img_patch_2 = img[0:64, 64:224, :]
    img_patch_3 = img[64:224, 0:64, :]
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

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
        transforms.ToTensor(),
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
        transforms.ToTensor(),
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
        transforms.ToTensor(),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img_patch_1 = transforms.ToPILImage()(pipeline_montage_1(img_patch_1))
    img_patch_2 = transforms.ToPILImage()(pipeline_montage_23(img_patch_2))  # tran
    img_patch_3 = transforms.ToPILImage()(pipeline_montage_23(img_patch_3))
    img_patch_4 = transforms.ToPILImage()(pipeline_montage_4(img_patch_4))

    #  do not need shuffle
    img2[0:160, 0:64] = img_patch_3
    img2[0:160, 64:224] = img_patch_1
    img2[160:224,0:64] = img_patch_4
    img2[160:224, 64:224] = img_patch_2

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img2)

    # img2 = Image.fromarray(img2)
    # img2.show()

    return img2


def random_montage_size_4132(img):

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy
    img_patch_2 = img[0:64, 64:224, :]
    img_patch_3 = img[64:224, 0:64, :]
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

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
        transforms.ToTensor(),
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
        transforms.ToTensor(),
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
        transforms.ToTensor(),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img_patch_1 = transforms.ToPILImage()(pipeline_montage_1(img_patch_1))
    img_patch_2 = transforms.ToPILImage()(pipeline_montage_23(img_patch_2))  # tran
    img_patch_3 = transforms.ToPILImage()(pipeline_montage_23(img_patch_3))
    img_patch_4 = transforms.ToPILImage()(pipeline_montage_4(img_patch_4))

    #  do not need shuffle
    img2[0:64, 0:64] = img_patch_4
    img2[0:160, 64:224] = img_patch_1
    img2[64:224,0:64] = img_patch_3
    img2[160:224, 64:224] = img_patch_2

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img2)

    # img2 = Image.fromarray(img2)
    # img2.show()

    return img2


def random_montage_size_1432(img):

    # img = img.resize((224, 224), Image.ANTIALIAS)
    # img = np.array(img).astype(np.uint8)

    img_patch_1 = img[0:64, 0:64, :]  # 3 channel need to copy
    img_patch_2 = img[0:64, 64:224, :].transpose(1, 0, 2)
    img_patch_3 = img[64:224, 0:64, :].transpose(1, 0, 2)
    img_patch_4 = img[64:224, 64:224, :]

    ## neeed to tran to Image format
    img_patch_1 = Image.fromarray(img_patch_1, mode='RGB')
    img_patch_2 = Image.fromarray(img_patch_2, mode='RGB')
    img_patch_3 = Image.fromarray(img_patch_3, mode='RGB')
    img_patch_4 = Image.fromarray(img_patch_4, mode='RGB')

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
        transforms.ToTensor(),
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
        transforms.ToTensor(),
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
        transforms.ToTensor(),
    ])

    img2 = np.zeros((224, 224, 3), dtype=np.uint8)
    img_patch_1 = transforms.ToPILImage()(pipeline_montage_1(img_patch_1))
    img_patch_2 = transforms.ToPILImage()(pipeline_montage_23(img_patch_2))  # tran
    img_patch_3 = transforms.ToPILImage()(pipeline_montage_23(img_patch_3))
    img_patch_4 = transforms.ToPILImage()(pipeline_montage_4(img_patch_4))

    #  do not need shuffle
    img2[0:160, 0:160] = img_patch_1
    img2[0:64, 160:224] = img_patch_4
    img2[160:224,0:160] = img_patch_3
    img2[64:224, 160:224] = img_patch_2

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img2)
    #
    # img2 = Image.fromarray(img2)
    # img2.show()

    return img2


def random_choice(img):

    ## obtain img (PIL.image)

    random_order = random.randint(0, 7)
    img = img.resize((224, 224), Image.ANTIALIAS)
    img = np.array(img).astype(np.uint8)
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

def piltotensor(img):

    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img2 = totensor(img)

    return img2

# # test 为了画图
# #
# img_path = '/home/dataE/pycharmproject/why/paper_fig_example/montage_1_src.jpg'
# img = Image.open(img_path)
# img2 = v11_loc_trans(img)
# # # img = img.resize((224, 224), Image.ANTIALIAS)
# img1 = dataset_trans(img)
# img2 = montage_PIL(img)
# img3 = random_montage_size_1432(img)
# img4,_ = random_subfigure_cutout(img)
#
# ## just for visualize
# img1_unnor = UnNormalize()(img1)
# img2_unnor = UnNormalize()(img2)
# img3_unnor = UnNormalize()(img3)
# img4_unnor = UnNormalize()(img4)
#
# img1_unnor = torch.squeeze(img1_unnor, dim=0)
# img2_unnor = torch.squeeze(img2_unnor, dim=0)
# img3_unnor = torch.squeeze(img3_unnor, dim=0)
# img4_unnor = torch.squeeze(img4_unnor, dim=0)
#
# img1_PIL = transforms.ToPILImage()(img1_unnor)
# img2_PIL = transforms.ToPILImage()(img2_unnor)
# img3_PIL = transforms.ToPILImage()(img3_unnor)
# img4_PIL = transforms.ToPILImage()(img4_unnor)
#
# img1 = img1_PIL.save('/home/dataE/pycharmproject/why/paper_fig_example/img1_loc.jpg')
# img2 = img2_PIL.save('/home/dataE/pycharmproject/why/paper_fig_example/img2_loc.jpg')
# img3 = img3_PIL.save('/home/dataE/pycharmproject/why/paper_fig_example/img3_loc.jpg')
# img4 = img4_PIL.save('/home/dataE/pycharmproject/why/paper_fig_example/img4_loc.jpg')




# img3 = v5_global_trans(img)

# img = np.array(img).astype(np.uint8)
# cut_img = random_montage_size_1432(img)
# img2 = random_cutout_superpixel(img)
#
# img.show()
# img = img.resize((224, 224), Image.ANTIALIAS)
# img = np.array(img).astype(np.uint8)
# cut_img = random_montage_2143(img)
# cut_img_2 = random_montage_3412(img)


# # test

# #
# cut_img = cutout_center(img)
# print(cut_img)
# cut_img.show()

# img1,img2 = montage(img)
#
# print('img1')
# img1.show()
# # img1.show()
# print('img2')
# img2.show()

# img3 = transform_center(img)
# img3.show()

# img3 = cutout_subfigure_center(img)
# img3.show()

