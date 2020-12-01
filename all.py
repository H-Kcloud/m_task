# coding: utf-8
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image
import random
import os
import torch
from skimage import io
from DeepFakeMask import dfl_full, facehull, components, extended
import cv2
import numpy as np
from umeyama import umeyama
import cv2, yaml, os
from  albumentations  import (
    HorizontalFlip, CLAHE, Blur, OpticalDistortion ,PadIfNeeded,
     HueSaturationValue,ImageCompression,GaussianBlur,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomContrast, RandomBrightness, Flip, OneOf, Compose,FancyPCA,ShiftScaleRotate,ToSepia,ISONoise
) 
import json
import copy
from imgaug import augmenters as iaa
from skimage import transform as sktransform
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
    x1 = np.int(np.maximum(0, x1))
    x2 = np.int(np.minimum(w-1, x2))
    y1 = np.int(np.maximum(0, y1))
    y2 = np.int(np.minimum(h-1, y2 ))
    im = imgs[0]
    head_mask = np.zeros_like(im)
    head_mask[y1:y2, x1:x2,:] = 255
    return head_mask

def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)



# 圆脸mask
def get_face_mask_more(shape, landmarks, OVERLAY_POINTS):

    im = np.zeros(shape, dtype=np.float32)

    group = OVERLAY_POINTS
    draw_convex_hull(im, landmarks[group], color=255)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    return im




def my_aug(new):
# randomly downsample after BI pipeline
    
    if random.randint(0,1):
        aug_size = random.randint(64, 317)
        new = Image.fromarray(new)
        if random.randint(0, 1):
            new = new.resize((aug_size, aug_size), Image.BILINEAR)
        else:
            new = new.resize((aug_size, aug_size), Image.NEAREST)
        new = np.array(new)
        # random jpeg compression after BI pipeline，这里应该也可以使用png
    
    if random.randint(0, 1):
 #       if random.randint(0, 1):
#            new = cv2.GaussianBlur(new, (5, 5), 0)
        quality = random.randint(60,100)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]

        face_img_encode = cv2.imencode('.jpg', new, encode_param)[1]
        new = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)


    return new


def blendImages(src, dst, mask, featherAmount=0.2):
    maskIndices = np.where(mask != 0)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (
                1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    return composedImg


### wiffne
###########fs###################

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + JAW_POINTS)




def get_face_mask(im, landmarks, OVERLAY_POINTS):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    group = OVERLAY_POINTS
    draw_convex_hull(im, landmarks[group], color=255)

    im = np.array([im, im, im]).transpose((1, 2, 0))
    #    cv2.imwrite('srm/output2.png', im)
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) / 255
    #    cv2.imwrite('srm/output3.png', im)
    # im = (im> 0) * 1.0
    # im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                 c2.T - (s2 / s1) * R * c1.T)),
                      np.matrix([0., 0., 1.])])


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP)
    return output_im


# from color_transfer import color_transfer

def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1
    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))
##############################################################################
def name_resolve(path):
    name = os.path.splitext(os.path.basename(path))[0]
    vid_id, frame_id = name.split('_')[0:2]
    return vid_id, frame_id


def total_euclidean_distance(a, b):
    assert len(a.shape) == 2
    return np.sum(np.linalg.norm(a - b, axis=1))


def random_get_hull(landmark, img1):
    hull_type = random.choice([0, 1, 2, 3])
    if hull_type == 0:
        mask = dfl_full(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
        return mask / 255
    elif hull_type == 1:
        mask = extended(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
        return mask / 255
    elif hull_type == 2:
        mask = components(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
        return mask / 255
    elif hull_type == 3:
        mask = facehull(landmarks=landmark.astype('int32'), face=img1, channels=3).mask
        return mask / 255


def random_erode_dilate(mask, ksize=None):
    if random.random() > 0.5:
        if ksize is None:
            ksize = random.randint(1, 21)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8) * 255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.erode(mask, kernel, 1) / 255
    else:
        if ksize is None:
            ksize = random.randint(1, 5)
        if ksize % 2 == 0:
            ksize += 1
        mask = np.array(mask).astype(np.uint8) * 255
        kernel = np.ones((ksize, ksize), np.uint8)
        mask = cv2.dilate(mask, kernel, 1) / 255
    return mask


# borrow from https://github.com/MarekKowalski/FaceSwap
def blendImages_mask(src, dst, mask, featherAmount=0.2):
    maskIndices = np.where(mask != 0)

    src_mask = np.ones_like(mask)
    dst_mask = np.zeros_like(mask)

    maskPts = np.hstack((maskIndices[1][:, np.newaxis], maskIndices[0][:, np.newaxis]))
    faceSize = np.max(maskPts, axis=0) - np.min(maskPts, axis=0)
    featherAmount = featherAmount * np.max(faceSize)

    hull = cv2.convexHull(maskPts)
    dists = np.zeros(maskPts.shape[0])
    for i in range(maskPts.shape[0]):
        dists[i] = cv2.pointPolygonTest(hull, (maskPts[i, 0], maskPts[i, 1]), True)

    weights = np.clip(dists / featherAmount, 0, 1)

    composedImg = np.copy(dst)
    composedImg[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src[maskIndices[0], maskIndices[1]] + (
                1 - weights[:, np.newaxis]) * dst[maskIndices[0], maskIndices[1]]

    composedMask = np.copy(dst_mask)
    composedMask[maskIndices[0], maskIndices[1]] = weights[:, np.newaxis] * src_mask[maskIndices[0], maskIndices[1]] + (
            1 - weights[:, np.newaxis]) * dst_mask[maskIndices[0], maskIndices[1]]

    return composedImg, composedMask



# borrow from https://github.com/MarekKowalski/FaceSwap
def colorTransfer(src, dst, mask):
    transferredDst = np.copy(dst)

    maskIndices = np.where(mask != 0)

    maskedSrc = src[maskIndices[0], maskIndices[1]].astype(np.int32)
    maskedDst = dst[maskIndices[0], maskIndices[1]].astype(np.int32)

    meanSrc = np.mean(maskedSrc, axis=0)
    meanDst = np.mean(maskedDst, axis=0)

    maskedDst = maskedDst - meanDst
    maskedDst = maskedDst + meanSrc
    maskedDst = np.clip(maskedDst, 0, 255)

    transferredDst[maskIndices[0], maskIndices[1]] = maskedDst

    return transferredDst

def blend(img_copy, new, mask, blend_method):
    if blend_method == 0:
        new = (mask) * new + (1 - mask) * img_copy

    elif blend_method == 1:
        mask_255 = (mask * 255).astype(np.uint8)
        r = cv2.boundingRect(mask_255[:, :, 0])
        center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
        # Clone seamlessly.  提供了两种方式 cv2.MIXED_CLONE 和 cv2.NORMAL_CLONE 结果不同的
        new = cv2.seamlessClone(new, img_copy, mask_255, center, flags=cv2.NORMAL_CLONE)

    elif blend_method == 2:
        new = blendImages(new, img_copy, mask * 255)
    return new

def deblend(img_copy, new, mask, blend_method):
    if blend_method == 0:
        new = (1 - mask) * new + (mask) * img_copy

    elif blend_method == 1:
        mask_255 = (mask * 255).astype(np.uint8)
        r = cv2.boundingRect(mask_255[:, :, 0])
        center = (r[0] + int(r[2] / 2), r[1] + int(r[3] / 2))
        # Clone seamlessly.  提供了两种方式 cv2.MIXED_CLONE 和 cv2.NORMAL_CLONE 结果不同的
        new = cv2.seamlessClone(img_copy, new, mask_255, center, flags=cv2.NORMAL_CLONE)

    elif blend_method == 2:
        new = blendImages(img_copy, new, mask * 255)
    return new

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
        with open('point.json', 'r') as f:
            self.dict = json.load( f)

        with open('point.json', 'r') as f:
            self.landmarks_record = json.load(f)
            for k, v in self.landmarks_record.items():
                self.landmarks_record[k] = np.array(v)

        with open('found_video.json', 'r') as f:
            self.video_record = json.load(f)

        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.1))])
        self.augMethod = 1


    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = cv2.imread(fn)     #  bgr格式,像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        #print(fn)
        img_copy = copy.deepcopy(img)
        # 做替换
     #   print(fn)
        #im = cv2.imread('baseline/MM.png')
        #print(img)
        # 圆脸， 三角脸， 下半脸，
        OVERLAY_POINTS_list = [
            LEFT_BROW_POINTS + RIGHT_BROW_POINTS + JAW_POINTS,
            LEFT_BROW_POINTS + RIGHT_BROW_POINTS + [48, 59, 58, 57, 56, 55, 54],
            NOSE_POINTS + list(range(1, 16)),
            [29] + list(range(2, 15)) ,
            MOUTH_POINTS+list(range(4, 13))
            ]

        fn_p = fn.split('/')[-1]
        
        point = np.array(self.dict[fn_p])
        # 选择mask，

       # with open('found_video.json', 'r') as f:
        #    self.video_record = json.load(f)

        mask_index = random.choice([0,1,2,3,4,5])
        if mask_index != 5:
            OVERLAY_POINTS = OVERLAY_POINTS_list[mask_index]

            mask = get_face_mask_more(img.shape[0:2], point, OVERLAY_POINTS)
        elif mask_index == 5:
            mask = cut_head([img], point)
     
        '''
        OVERLAY_POINTS = OVERLAY_POINTS_list[0]
        mask = get_face_mask_more(img.shape[0:2], point, OVERLAY_POINTS)
        '''
        #c = random.choice([0,1,2,3,4,    5,6,7,8,9,10,11,12,13,14,15])
        

        blend_method = random.choice([0,2])
     
        #print(fn_p)
  
         

        if random.choice([0,1]):
            new = img
            label = 0

        else:
            mask = mask / 255
            mask = mask.astype(np.float32)
            mask = self.distortion.augment_image(mask)
            mask = random_erode_dilate(mask)
            if np.sum(mask) == 0 or np.sum(1 - mask) == 0:
                raise NotImplementedError

            if  self.augMethod == 1 : #f1
                # 正向高斯平滑
                label = 1
                #print('gsdata')
                siz = random.choice([7, 9, 11])
                new = cv2.GaussianBlur(img_copy, (siz, siz), 0)
                new = blend(img_copy, new, mask, blend_method)
                

            elif self.augMethod  == 2: #f2
                label = 1
                #print('De_gsdata')
                siz = random.choice([7, 9, 11])
                new = cv2.GaussianBlur(img_copy, (siz, siz), 0)
                new = deblend(img_copy, new, mask, blend_method)

            elif self.augMethod == 3 : #f1
                # 正向scaling
                label = 1
                
                siz = random.randint(64, 128)
                H, W = img.shape[0:2]
                resized = cv2.resize(img_copy, (int(siz), int(siz)))
                new = cv2.resize(resized, (W, H),  cv2.INTER_NEAREST)

                new = blend(img_copy, new, mask, blend_method)

            elif self.augMethod  == 4: #f2
                label = 1
                siz = random.randint(64, 128)
                H, W = img.shape[0:2]
                resized = cv2.resize(img_copy, (int(siz), int(siz)))
                new = cv2.resize(resized, (W, H),  cv2.INTER_NEAREST)
                new = deblend(img_copy, new, mask, blend_method)

            elif self.augMethod == 5 : #f1
                # 正向scaling
                label = 1
                
                aug = ISONoise(color_shift=(0.08, 0.15), intensity=(0.1, 0.2), p=1)
                new = aug(image=img_copy)['image']


                new = blend(img_copy, new, mask, blend_method)

            elif self.augMethod  == 6: #f2
                label = 1
                aug = ISONoise(color_shift=(0.08, 0.15), intensity=(0.1, 0.2), p=1)
                new = aug(image=img_copy)['image']

                new = deblend(img_copy, new, mask, blend_method)

            elif self.augMethod == 7 : #f1
                # 正向scaling
                label = 1
                siz = random.uniform(3, 4)
                if random.choice([0,1]):
                    siz = -siz


                aug = ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=(siz, siz), p=1)
                new = aug(image=img_copy)['image']

                new = blend(img_copy, new, mask, blend_method)

            elif self.augMethod  == 8: #f2
                

                f1, label = self.imgs[index]
                f2 = str(random.sample(self.dict.keys(), 1)[0])
                f1_read =  f1
                label = 1
                im1 = cv2.imread(f1_read)
                landmarks1 = np.array(self.dict[f1.split('/')[-1]])

                landmarks2 = np.array(self.dict[f2])
                f2 = 'train/resize_image/' + f2
                im2 = cv2.imread(f2)

                landmarks1 = np.matrix([[p[0], p[1]] for p in landmarks1])
                landmarks2 = np.matrix([[p[0], p[1]] for p in landmarks2])


                M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

                mask_index = random.choice([0, 1, 2, 3, 4])
                OVERLAY_POINTS = OVERLAY_POINTS_list[mask_index]

                mask_swap = get_face_mask(im2, landmarks2, OVERLAY_POINTS)

                warped_mask = warp_im(mask_swap, M, im1.shape)
                mask = np.max([get_face_mask(im1, landmarks1, OVERLAY_POINTS), warped_mask],
                      axis=0)


                warped_im2 = warp_im(im2, M, im1.shape)

                warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
                warped_corrected_im2 = np.around(warped_corrected_im2)
                warped_corrected_im2[warped_corrected_im2 > 255] = 255
                warped_corrected_im2[warped_corrected_im2 < 0] = 0  # saturation
                warped_corrected_im2 = warped_corrected_im2.astype(np.uint8)


                new = blend(im1, warped_corrected_im2, mask, blend_method)

            elif self.augMethod == 0:  # F的数据
                label = 1
                #print('alldata')
                t = random.choice([0,1,2,3,4])
                if t == 0: #gs,degs
                    siz = random.choice([7, 9, 11])
                    new = cv2.GaussianBlur(img_copy, (siz, siz), 0)
                    if random.choice([0,1]):
                        new = blend(img_copy, new, mask, blend_method)
                    else:
                        new = deblend(img_copy, new, mask, blend_method)
                elif t == 1 :
                    siz = random.randint(64, 128)
                    H, W = img.shape[0:2]
                    resized = cv2.resize(img_copy, (int(siz), int(siz)))
                    new = cv2.resize(resized, (W, H),  cv2.INTER_NEAREST)
                    if random.choice([0,1]):
                        new = blend(img_copy, new, mask, blend_method)
                    else:
                        new = deblend(img_copy, new, mask, blend_method)
                elif t == 2 :
                    aug = ISONoise(color_shift=(0.08, 0.15), intensity=(0.1, 0.2), p=1)
                    new = aug(image=img_copy)['image']
                    if random.choice([0,1]):
                        new = blend(img_copy, new, mask, blend_method)
                    else:
                        new = deblend(img_copy, new, mask, blend_method)

                elif t == 3 :
                    siz = random.uniform(3, 4)
                    if random.choice([0,1]):
                        siz = -siz
                    aug = ShiftScaleRotate(shift_limit=0, scale_limit=0, rotate_limit=(siz, siz), p=1)
                    new = aug(image=img_copy)['image']
                    new = blend(img_copy, new, mask, blend_method)

                elif t == 4 :
                    f1, label = self.imgs[index]
                    f2 = str(random.sample(self.dict.keys(), 1)[0])
                    f1_read =  f1
                    label = 1
                    im1 = cv2.imread(f1_read)
                    landmarks1 = np.array(self.dict[f1.split('/')[-1]])

                    landmarks2 = np.array(self.dict[f2])
                    f2 = 'train/resize_image/' + f2
                    im2 = cv2.imread(f2)

                    landmarks1 = np.matrix([[p[0], p[1]] for p in landmarks1])
                    landmarks2 = np.matrix([[p[0], p[1]] for p in landmarks2])


                    M = transformation_from_points(landmarks1[ALIGN_POINTS], landmarks2[ALIGN_POINTS])

                    mask_index = random.choice([0, 1, 2, 3, 4])
                    OVERLAY_POINTS = OVERLAY_POINTS_list[mask_index]

                    mask_swap = get_face_mask(im2, landmarks2, OVERLAY_POINTS)

                    warped_mask = warp_im(mask_swap, M, im1.shape)
                    mask = np.max([get_face_mask(im1, landmarks1, OVERLAY_POINTS), warped_mask],
                      axis=0)


                    warped_im2 = warp_im(im2, M, im1.shape)

                    warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
                    warped_corrected_im2 = np.around(warped_corrected_im2)
                    warped_corrected_im2[warped_corrected_im2 > 255] = 255
                    warped_corrected_im2[warped_corrected_im2 < 0] = 0  # saturation
                    warped_corrected_im2 = warped_corrected_im2.astype(np.uint8)


                    new = blend(im1, warped_corrected_im2, mask, blend_method)
                


        new = new.astype(np.uint8)

        new = my_aug(new)

        lin = Image.fromarray(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))
        if self.transform is not None:
            new = self.transform(lin)  # 在这里做transform，转为tensor等等


        return new, label
  
   


    def get_blended_face(self, background_face_path):

        # 补全路径
        background_face_filename = background_face_path
        background_face_path =  'train/resize_image/' + background_face_path

        background_face = io.imread(background_face_path)
        background_landmark = self.landmarks_record[background_face_filename]
        foreground_face = background_face
        #rows, cols = foreground_face.shape[:2]
# x轴平移100，y轴平移
        #xx = random.choice([5,10,-5,-10])
        #MMM = np.float32([[1, 0, xx], [0, 1, xx]])
# 用仿射变换实现平移，第三个参数为dst的大小
        #foreground_face = cv2.warpAffine(foreground_face, MMM, (cols, rows))

        # 找近邻图像，我已经找好存入了json文件中，只需随机选择一个即可
        foreground_face_filename = random.choice(self.video_record[background_face_filename])
        foreground_face_path = 'train/resize_image/' + foreground_face_filename
        # print('fore:', foreground_face_path)
        # foreground_face_path = self.search_similar_face(background_landmark,background_face_path)
        foreground_face = io.imread(foreground_face_path)

        # down sample before blending
        aug_size = random.randint(128, 317)
        background_landmark = background_landmark * (aug_size / 317)
        foreground_face = sktransform.resize(foreground_face, (aug_size, aug_size), preserve_range=True).astype(
            np.uint8)
        background_face = sktransform.resize(background_face, (aug_size, aug_size), preserve_range=True).astype(
            np.uint8)

        # get random type of initial blending mask 获得初始化mask
        mask = random_get_hull(background_landmark, background_face)

        #  random deform mask，随机变形mask之后，再随机腐蚀或膨胀
        mask = self.distortion.augment_image(mask)
        mask = random_erode_dilate(mask)
        # filte empty mask after deformation
        if np.sum(mask) == 0:
            print('============================')
            raise NotImplementedError

        # apply color transfer
        foreground_face = colorTransfer(background_face, foreground_face, mask * 255)

        # blend two face
        #  print(foreground_face.shape, background_face.shape, mask.shape)
        blended_face, mask = blendImages_mask(foreground_face, background_face, mask * 255)
        blended_face = blended_face.astype(np.uint8)

        # resize back to default resolution
        blended_face = sktransform.resize(blended_face, (317, 317), preserve_range=True).astype(np.uint8)
        mask = sktransform.resize(mask, (317, 317), preserve_range=True)
        mask = mask[:, :, 0:1]
        return blended_face, mask

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
        
        #img = Image.open(fn).convert('RGB')     # 像素值 0~255，在transfrom.totensor会除以255，使像素值变成 0~1
        img = cv2.imread(fn)
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)


