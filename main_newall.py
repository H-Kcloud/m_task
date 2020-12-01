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


train_txt_path = 'txt/train.txt'

valid_txt_path_df = 'txt/val_df.txt'
valid_txt_path_fs = 'txt/val_fs.txt'
valid_txt_path_ff = 'txt/val_ff.txt'
valid_txt_path_nt = 'txt/val_nt.txt'
valid_txt_path_ori = 'txt/val_ori.txt'
#valid_txt_path_celeb = 'celeb_all/test.txt'
valid_txt_path_celeb = 'txt/celeb0.txt'
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
train_data = MyDataset(txt_path=train_txt_path, transform=trainTransform)
valid_data_df = Mytest(txt_path=valid_txt_path_df, transform=validTransform)
valid_data_fs = Mytest(txt_path=valid_txt_path_fs, transform=validTransform)
valid_data_ff = Mytest(txt_path=valid_txt_path_ff, transform=validTransform)
valid_data_nt = Mytest(txt_path=valid_txt_path_nt, transform=validTransform)
valid_data_ori = Mytest(txt_path=valid_txt_path_ori, transform=validTransform)
valid_data_celeb = Mytest(txt_path=valid_txt_path_celeb, transform=validTransform)

# 构建DataLoder
train_loader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True, num_workers=8)
valid_loader_df = DataLoader(dataset=valid_data_df, batch_size=valid_bs, num_workers=8)
valid_loader_fs = DataLoader(dataset=valid_data_fs, batch_size=valid_bs, num_workers=8)
valid_loader_ff = DataLoader(dataset=valid_data_ff, batch_size=valid_bs, num_workers=8)
valid_loader_nt = DataLoader(dataset=valid_data_nt, batch_size=valid_bs, num_workers=8)
valid_loader_ori = DataLoader(dataset=valid_data_ori, batch_size=valid_bs, num_workers=8)
valid_loader_celeb = DataLoader(dataset=valid_data_celeb, batch_size=valid_bs, num_workers=8)

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
#model_path = "/home/10701001/result/2class_all_8/xception_RAW.pkl"
model_path = None

if model_path is not None:
    model.load_state_dict(torch.load(model_path))
    print('Model found in {}'.format(model_path))
else:
    print('No model found, initializing random model.')
if 'cuda':
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


# ------------------------------------ step 3/5 : 定义损失函数和优化器 ------------------------------------

criterion = nn.CrossEntropyLoss()                                                # 选择损失函数
# setup optimizer
#params = filter(lambda p: p.requires_grad, model.parameters())
#optimizer = torch.optim.Adam(params, lr=1e-4)
#optimizer = optim.SGD(model.parameters(), lr=lr_init, momentum=0.9, dampening=0.1)    # 选择优化器
#optimizer = optim.Adam(params, lr=lr_init, betas=(0.1, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer1 = optim.Adam(itertools.chain(model.parameters(),fc1.parameters()), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer2 = optim.Adam(itertools.chain(model.parameters(),fc2.parameters()), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer3 = optim.Adam(itertools.chain(model.parameters(),fc3.parameters()), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer4 = optim.Adam(itertools.chain(model.parameters(),fc4.parameters()), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer5 = optim.Adam(itertools.chain(model.parameters(),fc5.parameters()), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer6 = optim.Adam(itertools.chain(model.parameters(),fc6.parameters()), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer7 = optim.Adam(itertools.chain(model.parameters(),fc7.parameters()), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)
optimizer8 = optim.Adam(itertools.chain(model.parameters(),fc8.parameters()), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)

optimizerF=optim.Adam(itertools.chain(model.parameters(),fc1.parameters(),fc2.parameters(),fc3.parameters(),fc4.parameters(),fc5.parameters(),fc6.parameters(),fc7.parameters(),fc8.parameters(),FC.parameters()), lr=lr_init, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=True)



#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.3)     # 设置学习率下降策略
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, /
#                                   threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
# 余弦模拟退火调整学习率
#scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer,T_0=10,T_mult=2,eta_min=0)


#warm up+cosine调整学习率
# 每步的学习率列表
#alpha_plan = lr_scheduler(lr_init,  max_epoch , warmup_end_epoch=warmup_epochs, mode='cosine')
# 动量mom1从epoch_decay_start开始衰减成mom2
#print(alpha_plan)
def adjust_learning_rate(optimizer, epoch):
    for param_group in optimizer.param_groups:
        param_group['lr'] = alpha_plan[epoch]
        
        #param_group['betas'] = (beta1_plan[epoch], 0.999)  # only change beta1
        
if os.path.exists('checkpoints2.pkl'):
    checkpoint = torch.load('checkpoints.pkl')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch']
    print('加载 epoch {} 成功！'.format(start_epoch))
else:
    start_epoch = 0
    print('无保存模型，将从头开始训练！')   

# ------------------------------------ step 4/5 : 训练 --------------------------------------------------
train_txt = result_dir + '/'+ model_name + '_' + method + '_train.txt'
val_txt_df = result_dir + '/'+ model_name + '_' + method + '_val_df.txt'
val_txt_fs = result_dir + '/'+ model_name + '_' + method + '_val_fs.txt'
val_txt_ff = result_dir + '/'+ model_name + '_' + method + '_val_ff.txt'
val_txt_nt = result_dir + '/'+ model_name + '_' + method + '_val_nt.txt'
val_txt_ori = result_dir + '/'+ model_name + '_' + method + '_val_ori.txt'
val_txt_celeb = result_dir + '/'+ model_name + '_' + method + '_val_celeb.txt'
with open(train_txt, "a")as f2:
    with open(val_txt_df, "a")as f_df:
        with open(val_txt_fs, "a")as f_fs:
            with open(val_txt_ff, "a")as f_ff:
                with open(val_txt_nt, "a")as f_nt:
                    with open(val_txt_ori, "a")as f_ori:
                        with open(val_txt_celeb, "a")as f_celeb:
                            val_step = 1
                            now_cor_df = 50
                            now_cor_fs = 50
                            now_cor_ff = 50
                            now_cor_nt = 50
                            now_cor_ori = 50
                            now_cor_all = 100
                            now_cor_all2 = 100

                            for ite in range(0,99):  # 10ite
# ====================================小循环训练fc=======================
                 # iter= 10* 2 + 5 
                                print('-----------------start training----------------')
                                train_data.augMethod = 1
                                print('train fc1......fc9')
                                for epoch in range(0,80): # train fc1......fc9 80* 16+10*16
                                    length = len(train_loader)
                                
                                    loss_sigma_train = 0.0    # 记录一个epoch的loss之和
                                    correct_train = 0.0
                                    total_train = 0.0
                    #           adjust_learning_rate(optimizer, epoch)
                                    #print('-----------------start training----------------')
                                    model.train()
                                    fc1.train()
                                    fc2.train()
                                    fc3.train()
                                    fc4.train()
                                    fc5.train()
                                    fc6.train()
                                    fc7.train()
                                    fc8.train()
                                    FC.train()
                                    #print('augMethod', train_data.augMethod)
                                    for i, data in enumerate(train_loader):
                                   # model.train()
                                    
                                    # if i == 30 : break
                                    # 获取图片和标签  data为一个列表，第一个元素shape 为(64,3,32,32)，第二个元素是标签 shape 为(64)                                   #============================================================================================
                                        inputs, labels = data                                 #=======================================================================================

                                        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                                    # forward, backward, update weights
                                        if train_data.augMethod == 1:
                                            print('train fc1')
                                            optimizer1.zero_grad()
                                            outputs = model(inputs)
                                            outputs, xc = fc1(outputs)
                                        
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer1.step()
                                            train_data.augMethod = 2
                                        
                                        elif train_data.augMethod == 2:
                                            print('train fc2')
                                            optimizer2.zero_grad()
                                            outputs = model(inputs)
                                            outputs, xc = fc2(outputs)
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer2.step()
                                            train_data.augMethod = 3

                                        elif train_data.augMethod == 3:
                                            print('train fc3')
                                            optimizer3.zero_grad()
                                            outputs = model(inputs)
                                            outputs, xc = fc3(outputs)
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer3.step()
                                            train_data.augMethod =4

 
                                        elif train_data.augMethod == 4:
                                            print('train fc4')
                                            optimizer4.zero_grad()
                                            outputs = model(inputs)
                                            outputs, xc = fc4(outputs)
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer4.step()
                                            train_data.augMethod = 5
                                        elif train_data.augMethod == 5:
                                            print('train fc5')
                                            optimizer5.zero_grad()
                                            outputs = model(inputs)
                                            outputs, xc = fc5(outputs)
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer5.step()
                                            train_data.augMethod = 6

                                        elif train_data.augMethod == 6:
                                            print('train fc6')
                                            optimizer6.zero_grad()
                                            outputs = model(inputs)
                                            outputs, xc = fc6(outputs)
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer6.step()
                                            train_data.augMethod = 7

 
                                        elif train_data.augMethod == 7:
                                            print('train fc7')
                                            optimizer7.zero_grad()
                                            outputs = model(inputs)
                                            outputs, xc = fc7(outputs)
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer7.step()
                                            train_data.augMethod = 8 
                                        elif train_data.augMethod == 8:
                                            print('train fc8')
                                            optimizer8.zero_grad()
                                            outputs = model(inputs)
                                            outputs, xc = fc8(outputs)
                                            loss = criterion(outputs, labels)
                                            loss.backward()
                                            optimizer8.step()
                                            train_data.augMethod = 1                                     
                                        break
 

                                print('end fc1......fc9')

# ============================================================================
                                train_data.augMethod = 0 
                                loss_sigma_train = 0.0    # 记录一个epoch的loss之和
                                correct_train = 0.0
                                total_train = 0.0
                                #print('-----------------start training----------------')

                                
                                #print('augMethod', train_data.augMethod)
                                for i, data in enumerate(train_loader):
                                   
                                    length = len(train_loader)                                 #============================================================================================
                                    inputs, labels = data                                 #=======================================================================================
                                    inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
                                    # forward, backward, update weights
                                    if train_data.augMethod == 0:
                                        print('train FC')
                                        optimizerF.zero_grad()
                                        xce_feature = model(inputs)
                                        xc1, xf1 = fc1(xce_feature)
                                        xc2, xf2 = fc2(xce_feature)
                                        xc3, xf3 = fc3(xce_feature) 
                                        xc4, xf4 = fc4(xce_feature) 
                                        xc5, xf5 = fc5(xce_feature)
                                        xc6, xf6 = fc6(xce_feature)
                                        xc7, xf7 = fc7(xce_feature) 
                                        xc8, xf8 = fc8(xce_feature)                                    
                                        outputs = torch.cat([xf1,xf2,xf3,xf4,xf5,xf6,xf7,xf8], 1)
                                        
                                        outputs = FC(outputs)
                                        loss = criterion(outputs, labels)
                                        loss.backward()
                                        optimizerF.step()

                                        _, predicted = torch.max(outputs.data, 1)
                                        total_train += labels.size(0)

                                        correct_train += (predicted == labels).squeeze().sum().cpu().numpy()
                                        loss_sigma_train += loss.item()

                                    if i == 9:           #     训练10次 
                                        loss_avg = loss_sigma_train / 10
                                        loss_sigma_train = 0.0
                                        print('correct',correct_train,'total',total_train)
                                        print("ite: {} Loss: {:.4f} Acc:{:.2%}".format(
                                            ite, loss_avg, correct_train / total_train))
                                        f2.write("ite: {} Loss: {:.4f} Acc:{:.2%}".format(ite, loss_avg, correct_train / total_train))
                                        f2.write('\n')
                                        f2.flush()
                                      
                                        break
 

                                print('end FC')
                        
                                if ite %10== 9:           #    
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
                                    length = len(valid_loader_df) # batch总数
                                    with torch.no_grad():
                                        for i, data in enumerate(valid_loader_df):
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
                                                              
                                            outputs = torch.cat([xf1,xf2,xf3,xf4,xf5,xf6,xf7,xf8], 1)
                                       
                                            outputs = FC(outputs)

                                            # val loss 
                                            loss = criterion(outputs, labels)
                                            loss_sigma_val += loss.item() 
                                            probability = F.softmax(outputs, dim=1)
              
                                            _, predicted = torch.max(probability, 1)
                                         
                                            predicted = torch.where(predicted > 0, torch.tensor(1).cuda(),      torch.tensor(0).cuda())
                                          
                                            total += labels.size(0)
                                            correct += (predicted == labels).squeeze().sum().cpu().numpy()
                                            
                                            for i in range(len(labels)):
                                                true_i = labels[i].cpu().numpy()
                                                pre_i = predicted[i].cpu().numpy()
                                                conf_mat[true_i, pre_i] += 1.0

                                    print('{} set {},Accuracy:{:.3%}'.format('valid', total, correct / total))
                                    print(conf_mat)
                                    print('0召回率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1]), '\n', \
                                              '0查准率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0]), '\n', \
                                              '1召回率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0]), '\n', \
                                              '1查准率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1]), '\n', \
                                              )
                                    if (100. * correct / total) > now_cor_df:
                                        print('better model is found')
                                        now_cor_df = 100. * correct / total

                                    f_df.write('%03d   | Acc: %.3f%% | loss: %.5f' % (val_step, 100. * correct / total , loss_sigma_val/length))
                                    f_df.write('\n')
                                    f_df.flush()
                                    df_v = 100. * correct / total

                                    print('=================faceswap验证开始==============')
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
                                    length = len(valid_loader_fs) # batch总数
                                    with torch.no_grad():
                                        for i, data in enumerate(valid_loader_fs):
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
                                                              
                                            outputs = torch.cat([xf1,xf2,xf3,xf4,xf5,xf6,xf7,xf8], 1)
                                       
                                            outputs = FC(outputs)

                                            # val loss 
                                            loss = criterion(outputs, labels)
                                            loss_sigma_val += loss.item() 
                                            probability = F.softmax(outputs, dim=1)
              
                                            _, predicted = torch.max(probability, 1)
                                         
                                            predicted = torch.where(predicted > 0, torch.tensor(1).cuda(),      torch.tensor(0).cuda())
                                          
                                            total += labels.size(0)
                                            correct += (predicted == labels).squeeze().sum().cpu().numpy()
                                            
                                            for i in range(len(labels)):
                                                true_i = labels[i].cpu().numpy()
                                                pre_i = predicted[i].cpu().numpy()
                                                conf_mat[true_i, pre_i] += 1.0

                                    print('{} set {},Accuracy:{:.3%}'.format('valid', total, correct / total))
                                    print(conf_mat)
                                    print('0召回率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1]), '\n', \
                                              '0查准率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0]), '\n', \
                                              '1召回率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0]), '\n', \
                                              '1查准率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1]), '\n', \
                                              )
                                    if (100. * correct / total) > now_cor_fs:
                                        print('better model is found')
                                        now_cor_fs = 100. * correct / total

                                    f_fs.write('%03d   | Acc: %.3f%% | loss: %.5f' % (val_step, 100. * correct / total , loss_sigma_val/length))
                                    
                                    f_fs.write('\n')
                                    f_fs.flush()
                                    fs_v = 100. * correct / total
 
                                    print('=================face2face验证开始==============')
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
                                    length = len(valid_loader_ff) # batch总数
                                    with torch.no_grad():
                                        for i, data in enumerate(valid_loader_ff):
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
                                                              
                                            outputs = torch.cat([xf1,xf2,xf3,xf4,xf5,xf6,xf7,xf8], 1)
                                       
                                            outputs = FC(outputs)

                                            # val loss 
                                            loss = criterion(outputs, labels)
                                            loss_sigma_val += loss.item() 
                                            probability = F.softmax(outputs, dim=1)
              
                                            _, predicted = torch.max(probability, 1)
                                         
                                            predicted = torch.where(predicted > 0, torch.tensor(1).cuda(),      torch.tensor(0).cuda())
                                          
                                            total += labels.size(0)
                                            correct += (predicted == labels).squeeze().sum().cpu().numpy()
                                            
                                            for i in range(len(labels)):
                                                true_i = labels[i].cpu().numpy()
                                                pre_i = predicted[i].cpu().numpy()
                                                conf_mat[true_i, pre_i] += 1.0

                                    print('{} set {},Accuracy:{:.3%}'.format('valid', total, correct / total))
                                    print(conf_mat)
                                    print('0召回率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1]), '\n', \
                                              '0查准率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0]), '\n', \
                                              '1召回率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0]), '\n', \
                                              '1查准率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1]), '\n', \
                                              )
                                    if (100. * correct / total) > now_cor_ff:
                                        print('better model is found')
                                        now_cor_ff = 100. * correct / total

                                    f_ff.write('%03d   | Acc: %.3f%% | loss: %.5f' % (val_step, 100. * correct / total , loss_sigma_val/length))
                                    f_ff.write('\n')
                                    f_ff.flush()
                                    ff_v = 100. * correct / total

                                    print('=================NT验证开始==============')
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
                                    length = len(valid_loader_nt) # batch总数
                                    with torch.no_grad():
                                        for i, data in enumerate(valid_loader_nt):
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
                                                              
                                            outputs = torch.cat([xf1,xf2,xf3,xf4,xf5,xf6,xf7,xf8], 1)
                                       
                                            outputs = FC(outputs)

                                            # val loss 
                                            loss = criterion(outputs, labels)
                                            loss_sigma_val += loss.item() 
                                            probability = F.softmax(outputs, dim=1)
              
                                            _, predicted = torch.max(probability, 1)
                                         
                                            predicted = torch.where(predicted > 0, torch.tensor(1).cuda(),      torch.tensor(0).cuda())
                                          
                                            total += labels.size(0)
                                            correct += (predicted == labels).squeeze().sum().cpu().numpy()
                                            
                                            for i in range(len(labels)):
                                                true_i = labels[i].cpu().numpy()
                                                pre_i = predicted[i].cpu().numpy()
                                                conf_mat[true_i, pre_i] += 1.0

                                    print('{} set {},Accuracy:{:.3%}'.format('valid', total, correct / total))
                                    print(conf_mat)
                                    print('0召回率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1]), '\n', \
                                              '0查准率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0]), '\n', \
                                              '1召回率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0]), '\n', \
                                              '1查准率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1]), '\n', \
                                              )
                                    if (100. * correct / total) > now_cor_nt:
                                        print('better model is found')
                                        now_cor_nt = 100. * correct / total

                                    f_nt.write('%03d   | Acc: %.3f%% | loss: %.5f' % (val_step, 100. * correct / total, loss_sigma_val/length))
                                    f_nt.write('\n')
                                    f_nt.flush()
                                    nt_v = 100. * correct / total

                                    print('=================ori验证开始==============')
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
                                    length = len(valid_loader_ori) # batch总数
                                    with torch.no_grad():
                                        for i, data in enumerate(valid_loader_ori):
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
                                                              
                                            outputs = torch.cat([xf1,xf2,xf3,xf4,xf5,xf6,xf7,xf8], 1)
                                       
                                            outputs = FC(outputs)

                                            # val loss 
                                            loss = criterion(outputs, labels)
                                            loss_sigma_val += loss.item() 
                                            probability = F.softmax(outputs, dim=1)
              
                                            _, predicted = torch.max(probability, 1)
                                         
                                            predicted = torch.where(predicted > 0, torch.tensor(1).cuda(),      torch.tensor(0).cuda())
                                          
                                            total += labels.size(0)
                                            correct += (predicted == labels).squeeze().sum().cpu().numpy()
                                            
                                            for i in range(len(labels)):
                                                true_i = labels[i].cpu().numpy()
                                                pre_i = predicted[i].cpu().numpy()
                                                conf_mat[true_i, pre_i] += 1.0

                                    print('{} set {},Accuracy:{:.3%}'.format('valid', total, correct / total))
                                    print(conf_mat)
                                    print('0召回率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[0][1]), '\n', \
                                              '0查准率：', conf_mat[0][0] / (conf_mat[0][0] + conf_mat[1][0]), '\n', \
                                              '1召回率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[1][0]), '\n', \
                                              '1查准率：', conf_mat[1][1] / (conf_mat[1][1] + conf_mat[0][1]), '\n', \
                                              )
                                    if (100. * correct / total) > now_cor_ori:
                                        print('better model is found')
                                        now_cor_ori = 100. * correct / total

                                    f_ori.write('%03d   | Acc: %.3f%% | loss: %.5f' % (val_step, 100. * correct / total, loss_sigma_val/length))
                                    f_ori.write('\n')
                                    f_ori.flush()
                                    ori_v = 100. * correct / total


                                    val_step += 1
                            print('Finished Training')

                # ------------------------------------ step5: 保存模型 并且绘制混淆矩阵图 ------------------------------------
net_save_path = os.path.join(log_dir, '{}_{}.pkl'.format(model_name, method))
torch.save(model.state_dict(), net_save_path)

net_save_path = os.path.join(log_dir, 'fc1_{}_{}.pkl'.format(model_name, method))
torch.save(fc1.state_dict(), net_save_path)
net_save_path = os.path.join(log_dir, 'fc2_{}_{}.pkl'.format(model_name, method))
torch.save(fc2.state_dict(), net_save_path)
net_save_path = os.path.join(log_dir, 'fc3_{}_{}.pkl'.format(model_name, method))
torch.save(fc3.state_dict(), net_save_path)
net_save_path = os.path.join(log_dir, 'fc4_{}_{}.pkl'.format(model_name, method))
torch.save(fc4.state_dict(), net_save_path)
net_save_path = os.path.join(log_dir, 'fc5_{}_{}.pkl'.format(model_name, method))
torch.save(fc5.state_dict(), net_save_path)
net_save_path = os.path.join(log_dir, 'fc6_{}_{}.pkl'.format(model_name, method))
torch.save(fc6.state_dict(), net_save_path)
net_save_path = os.path.join(log_dir, 'fc7_{}_{}.pkl'.format(model_name, method))
torch.save(fc7.state_dict(), net_save_path)
net_save_path = os.path.join(log_dir, 'fc8_{}_{}.pkl'.format(model_name, method))
torch.save(fc8.state_dict(), net_save_path)



net_save_path = os.path.join(log_dir, 'FC_{}_{}.pkl'.format(model_name, method))
torch.save(FC.state_dict(), net_save_path)




