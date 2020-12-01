# coding: utf-8
from torch.utils.data import Dataset
import dlib
from skimage import io
from skimage import transform as sktransform
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import random
from PIL import Image
from imgaug import augmenters as iaa
from DeepFakeMask import dfl_full, facehull, components, extended
import cv2
import tqdm



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
    if random.random() > 1:
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
def blendImages(src, dst, mask, featherAmount=0.2):
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


###########################
class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            words = line.split() # 返回一组数据和标签的列�?
            imgs.append((words[0], 0)) #合成一个元组添加到列表�?
        self.imgs = imgs        # 最主要就是要生成这个list�?然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

        with open('point.json', 'r') as f:
            self.landmarks_record = json.load(f)
            for k, v in self.landmarks_record.items():
                self.landmarks_record[k] = np.array(v)

        # extract all frame from all video in the name of {videoid}_{frameid}
        # 近邻搜索得到的结�?
        with open('found_video.json', 'r') as f:
            self.video_record = json.load(f)
        '''
        self.data_list = [
                    '000_0000.png',
                    '001_0000.png'      
                    ] * 10000
        '''
        # predefine mask distortion
#############该处有参数################################################################
        self.distortion = iaa.Sequential([iaa.PiecewiseAffine(scale=(0.01, 0.1))])
#######################################################################################

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        # img = cv2.imread(fn)     #  bgr格式,像素�?0~255，在transfrom.totensor会除�?55，使像素值变�?0~1
        #print(fn)

        # 这里就是每次的输入训练数�?        # background_face_path = random.choice(self.data_list)
        background_face_path = fn.split('/')[-1]
        #print(background_face_path)

        data_type = 'real' if random.randint(0, 1) else 'fake'
        if data_type == 'fake':
            face_img, mask = self.get_blended_face(background_face_path)
            mask = (1 - mask) * mask * 4
        else:
            background_face_path =  'train/resize_image/' + background_face_path
            face_img = io.imread(background_face_path)
            mask = np.zeros((317, 317, 1))



        # randomly downsample after BI pipeline
        if random.randint(0, 1):

###################################################有参数#####################
            aug_size = random.randint(64, 317)
            face_img = Image.fromarray(face_img)
            if random.randint(0, 1):
                face_img = face_img.resize((aug_size, aug_size), Image.BILINEAR)
            else:
                face_img = face_img.resize((aug_size, aug_size), Image.NEAREST)



            face_img = face_img.resize((317, 317), Image.BILINEAR)
            face_img = np.array(face_img)

        # random jpeg compression after BI pipeline
        
        if random.randint(0, 1):
#############################有参数###########################################
            quality = random.randint(60, 100)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            face_img_encode = cv2.imencode('.jpg', face_img, encode_param)[1]
            face_img = cv2.imdecode(face_img_encode, cv2.IMREAD_COLOR)


        # 对图像再次裁�?
        face_img = face_img[60:317, 30:287, :]
        mask = mask[60:317, 30:287, :]
        
        # random flip
        if random.randint(0, 1):
            face_img = np.flip(face_img, 1)
            mask = np.flip(mask, 1)

        if data_type == 'real':
            label = 0
        else:
            label = 1
        #new_path = data_path + '/BI.png'
        #io.imsave(new_path, face_img)
        #new = cv2.imread(new_path)
        #new = Image.fromarray(cv2.cvtColor(new, cv2.COLOR_BGR2RGB))
        lin = Image.fromarray(face_img)
        if self.transform is not None:
            new = self.transform(lin)  # 在这里做transform，转为tensor等等
        return new, label


        # 先有背景脸， 再搜索前�?
    def get_blended_face(self, background_face_path):

        # 补全路径
        background_face_filename = background_face_path
        background_face_path =  'train/resize_image/' + background_face_path

        background_face = io.imread(background_face_path)
        background_landmark = self.landmarks_record[background_face_filename]

        # 找近邻图像，我已经找好存入了json文件中，只需随机选择一个即�?
        foreground_face_filename = random.choice(self.video_record[background_face_filename])
        foreground_face_path = 'train/resize_image/' + foreground_face_filename
        #print('fore:', foreground_face_path)
        # foreground_face_path = self.search_similar_face(background_landmark,background_face_path)
        foreground_face = io.imread(foreground_face_path)

        # down sample before blending
        aug_size = random.randint(128, 317)
        background_landmark = background_landmark * (aug_size / 317)
        foreground_face = sktransform.resize(foreground_face, (aug_size, aug_size), preserve_range=True).astype(
            np.uint8)
        background_face = sktransform.resize(background_face, (aug_size, aug_size), preserve_range=True).astype(
            np.uint8)

        # get random type of initial blending mask
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
        blended_face, mask = blendImages(foreground_face, background_face, mask * 255)
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
            words = line.split() # 返回一组数据和标签的列�?
            imgs.append((words[0], int(words[1]))) #合成一个元组添加到列表�?
        self.imgs = imgs        # 最主要就是要生成这个list�?然后DataLoader中给index，通过getitem读取图片数据
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        fn, label = self.imgs[index]
       
#        print(fn)
        img = Image.open(fn).convert('RGB')     # 像素�?0~255，在transfrom.totensor会除�?55，使像素值变�?0~1

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等

        return img, label

    def __len__(self):
        return len(self.imgs)


