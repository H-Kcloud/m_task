# coding: utf-8
from sklearn.utils import shuffle
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import os
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lr import lr_scheduler
#from focal_loss import BCEFocalLoss
from sklearn import metrics
import itertools
from all import MyDataset, Mytest

#from tensorboardX import SummaryWriter
from datetime import datetime
from network.models import model_selection
from network.xception import ourfc
from network.xception import ourFC

#['df', 'ff', 'fs', 'nt']
method = 'RAW'
#'resnet101','xception',
model_name = 'xception'
# image_size=256,299
image_size = 299

import argparse
# description参数可以用于描述脚本的参数作用，默认为空
parser=argparse.ArgumentParser(description="A description of what the program does")
parser.add_argument('--valid','-v',required = True, help='valid data')
args=parser.parse_args()
print(args.valid)

valid_txt_path = args.valid

train_bs = 16
valid_bs = 16
lr_init = 0.0002 #0.003
max_epoch = 20
warmup_epochs = 5
# log
result_dir = os.path.join( "result/newmodel")


log_dir = os.path.join(result_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# ------------------------------------ step 1/5 : 加载数据------------------------------------

# 数据预处理设置
normMean = [0.5, 0.5, 0.5]
normStd = [0.5, 0.5, 0.5]
normTransform = transforms.Normalize(normMean, normStd)
trainTransform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomCrop(image_size), #, padding=10),
    
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.RandomRotation(15),
    #transforms.RandomChoice([transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]   ),
    transforms.RandomChoice([transforms.RandomGrayscale(), transforms.ColorJitter(0.5, 0.5, 0.5, 0.5)]  ),
    transforms.ToTensor(),
    normTransform
])

validTransform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    normTransform
])

# 构建MyDataset实例

valid_data = Mytest(txt_path=valid_txt_path, transform=validTransform)

# 构建DataLoder

valid_loader = DataLoader(dataset=valid_data, batch_size=valid_bs, num_workers=4)

# ------------------------------------ step 2/5 : 定义网络------------------------------------
model, *_ = model_selection(modelname=model_name, num_out_classes=2)
fc1  = ourfc(2)
#print(model)
fc2  = ourfc(2)
fc3  = ourfc(2)
fc4  = ourfc(2)
fc5  = ourfc(2)
fc6  = ourfc(2)
fc7  = ourfc(2)
fc8  = ourfc(2)


FC  = ourFC(2)

use_cuda = torch.cuda.is_available()
Device = torch.device('cuda' if use_cuda else 'cpu')
print(Device)


#========================
# Load model
model_path_model = "/home/user/result/newmodel/xception_RAW.pkl"

model_path_fc1 = "/home/user/result/newmodel/fc1_xception_RAW.pkl"
model_path_fc2 = "/home/user/result/newmodel/fc2_xception_RAW.pkl"
model_path_fc3 = "/home/user/result/newmodel/fc3_xception_RAW.pkl"
model_path_fc4 = "/home/user/result/newmodel/fc4_xception_RAW.pkl"
model_path_fc5 = "/home/user/result/newmodel/fc5_xception_RAW.pkl"
model_path_fc6 = "/home/user/result/newmodel/fc6_xception_RAW.pkl"
model_path_fc7 = "/home/user/result/newmodel/fc7_xception_RAW.pkl"
model_path_fc8 = "/home/user/result/newmodel/fc8_xception_RAW.pkl"
model_path_FC = "/home/user/result/newmodel/FC_xception_RAW.pkl"

if model_path_FC is not None:
    model.load_state_dict(torch.load(model_path_model))
    fc1.load_state_dict(torch.load(model_path_fc1))
    fc2.load_state_dict(torch.load(model_path_fc2))
    fc3.load_state_dict(torch.load(model_path_fc3))
    fc4.load_state_dict(torch.load(model_path_fc4))
    fc5.load_state_dict(torch.load(model_path_fc5))
    fc6.load_state_dict(torch.load(model_path_fc6))
    fc7.load_state_dict(torch.load(model_path_fc7))
    fc8.load_state_dict(torch.load(model_path_fc8))
    FC.load_state_dict(torch.load(model_path_FC))
    print('Model found in {}'.format(model_path_FC))
else:
    print('No model found, initializing random model.')
if use_cuda:
    print('cude is ok')
    model = model.cuda()
    fc1 = fc1.cuda()
    fc2 = fc2.cuda()
    fc3 = fc3.cuda()
    fc4 = fc4.cuda()
    fc5 = fc5.cuda()
    fc6 = fc6.cuda()
    fc7 = fc7.cuda()
    fc8 = fc8.cuda()
    FC = FC.cuda()
print('model is built')
# ===========================



# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

criterion = nn.CrossEntropyLoss()                                                # 选择损失函数
# setup optimizer

# ----------------------- 观察模型在deepfake上的表现 --------------------------
print('=================deepfake验证开始==============')
conf_mat = np.zeros([2, 2])

correct = 0.0
total = 0.0
loss_sigma_val = 0.0
model.eval()
fc1.eval()
fc2.eval()
fc3.eval()
fc4.eval()
fc5.eval()
fc6.eval()
fc7.eval()
fc8.eval()                              
FC.eval()




loss_sigma = 0.0
correct = 0.0
total = 0.0

test_y = []
prodict_prob_y = []
prob=[]
proba=[]



for i, data in enumerate(valid_loader):
    with torch.no_grad():
        # 获取图片和标签
        images, labels = data
        images, labels = Variable(images).cuda(), Variable(labels).cuda()

        # forward
        xce_feature = model(images)
        xc1, xf1 = fc1(xce_feature)
        xc2, xf2 = fc2(xce_feature)
        xc3, xf3 = fc3(xce_feature)
        xc4, xf4 = fc4(xce_feature)
        xc5, xf5 = fc5(xce_feature)
        xc6, xf6 = fc6(xce_feature)
        xc7, xf7 = fc7(xce_feature)
        xc8, xf8 = fc8(xce_feature)

        outputs = torch.cat([xf1, xf2, xf3, xf4, xf5, xf6, xf7, xf8], 1)

        outputs = FC(outputs)
        # outputs.detach_()
        probability = F.softmax(outputs, dim=1)

        _, predicted = torch.max(probability, 1)

        predicted = torch.where(predicted > 0, torch.tensor(1).cuda(), torch.tensor(0).cuda())
        # 计算loss

        # 统计
        # print(probability[:,0:4])
        probability[:, 0] = probability[:, 0]  # +probability[:,1]+probability[:,2]+probability[:,3] + probability[:,4]
        probability[:, 1] = 1 - probability[:, 0]

        probability = probability[:, 0:2]
        # print(sum(probability[:,0]/labels.size(0)))
        prob = prob + (probability[:, 0].cpu().numpy().tolist())
        proba = proba + (probability[:, 1].cpu().numpy().tolist())

        probability = probability[:, 1]
        # labels = labels.data    # Variable --> tensor
        prodict_prob_y = prodict_prob_y + (probability.cpu().numpy().tolist())
        test_y = test_y + (labels.cpu().numpy().tolist())
        total += labels.size(0)
        correct += (predicted == labels).squeeze().sum().cpu().numpy()
        for i in range(len(labels)):
            true_i = labels[i].cpu().numpy()
            pre_i = predicted[i].cpu().numpy()
            conf_mat[true_i, pre_i] += 1.0

print('0acc:', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1]))
print('1召回率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0]))
print('{} set {},Accuracy:{:.3%}'.format('valid', total, correct / total))
print(conf_mat)
print('0召回率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1]), '\n', \
      '0查准率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0]), '\n', \
      '1召回率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0]), '\n', \
      '1查准率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1]), '\n', \
      )
test_auc = metrics.roc_auc_score(test_y, prodict_prob_y)  # 验证集上的auc值
print('test_auc: ', test_auc)   





