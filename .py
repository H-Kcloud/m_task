# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image
import random

import cv2
import numpy as np
from umeyama import umeyama
import cv2, yaml, os, dlib
from  albumentations  import (
    HorizontalFlip, CLAHE, Blur, OpticalDistortion ,PadIfNeeded,
     HueSaturationValue,ImageCompression,GaussianBlur,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose,FancyPCA,ShiftScaleRotate,ToSepia,ISONoise
) 
import json
import copy

#from color_transfer import color_transfer

# 脸部51个点的平均坐标

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))



def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords



# 返回裁切人和边框坐标
def cut_head(imgs, point, seed=None):

    h, w = imgs[0].shape[:2]
    x1, y1 = np.min(point, axis=0)
    x2, y2 = np.max(point, axis=0)
    im = imgs[0]
    head_mask = np.zeros_like(im)
    head_mask[y1:y2, x1:x2,:] = 255
    return head_mask

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)



# 圆脸mask
def get_face_mask_more(shape, landmarks, OVERLAY_POINTS):

    im = np.zeros(shape, dtype=np.uint8)

    group = OVERLAY_POINTS
    draw_convex_hull(im, landmarks[group], color=255)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    return im


##############################

SCALE_FACTOR = 1 
FEATHER_AMOUNT = 11



#

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6
PREDICTOR_PATH = "ff++/baseline/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


###########################
class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split() # 返回一组数据和标签的列表
            imgs.append((words[0], 0)) #合成一个元组添加到列表中

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform
        #with open('last/point.json', 'r') as f:
        with open('ff++/point.json', 'r') as f:
            self.dict = json.load( f) 

      

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = cv2.imread(fn)     #  bgr格式,像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img_copy = copy.deepcopy(img)
        # 做替换
   
        #im = cv2.imread('baseline/MM.png')

        # 圆脸， 三角脸， 下半脸，
        OVERLAY_POINTS_list = [
            LEFT_BROW_POINTS + RIGHT_BROW_POINTS + [48, 59, 58, 57, 56, 55, 54],
            ]

        fn_p = fn.split('/')[-1]
        
        point = np.array(self.dict[fn_p])
        # 选择mask，
        mask_index = random.choice([0, 1])
        if mask_index != 1:
            OVERLAY_POINTS = OVERLAY_POINTS_list[mask_index]

            mask = get_face_mask_more(img.shape[0:2], point, OVERLAY_POINTS)
        elif mask_index == 1:
            mask = cut_head([img], point)

        c = random.choice([0,1])
        
        if c == 1 :

            siz = random.choice([3,5,7])
            # 边缘模糊差别不大
            mask = mask/255

            img = cv2.GaussianBlur(img, (siz, siz), 0)
            new = mask * img + (1 - mask) * img_copy

                #new = new.astype(np.uint8)
            cv2.imwrite('new.png', new)
            new = cv2.imread('new.png')
            label = 1
            

        
            
        else :
            siz = 0
            new = img
            label = 0
        


        cv2.imwrite('main3/new.png', new)
        new = cv2.imread('main3/new.png') 
        #cv2.imwrite('main3/{}.png'.format(label), new)
        new = Image.fromarray(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            new = self.transform(new)   # 在这里做transform，转为tensor等等

            
        

        return new, label

    def __len__(self):
        return len(self.imgs)

class Mytest(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split() # 返回一组数据和标签的列表
            imgs.append((words[0], int(words[1]))) #合成一个元组添加到列表中

        self.imgs = imgs        # 最主要就是要生成这个list， 然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)


