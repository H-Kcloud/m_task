"""

Author: Andreas Rössler
"""
import os
import argparse


import torch

import torch.nn as nn
import torch.nn.functional as F

from network.xception import xception
from network.hrnet import get_cls_net
import math
import torchvision


def return_pytorch04_xception(pretrained=False):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(pretrained='imagenet')
    if pretrained:
        # Load model in torch 0.4+
        model.fc = model.last_linear
        del model.last_linear
        
        state_dict = torch.load(
            'xception-b5690688.pth')
        
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        model.load_state_dict(state_dict)
        print('xception pretrained is found')
        model.last_linear = model.fc
        del model.fc
    return model
    
def return_mynet_xception():
    # Raises warning "src not broadcastable to dst" but thats fine
    model = mynet()
    return model
def return_hrnet(pretrained):
    # Raises warning "src not broadcastable to dst" but thats fine
    
    model = get_cls_net()
    if pretrained:
        # Load model in torch 0.4+
        #model.fc = model.last_linear
        #del model.last_linear
        model_dict = model.state_dict()
        pretrained_dict = torch.load(
            '/mnt/tangjinhui/10119_hongkai/ff_run/network/hrnetv2_w64_imagenet_pretrained.pth')

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新现有的model_dict
        model_dict.update(pretrained_dict)
        # 加载我们真正需要的state_dict
        model.load_state_dict(model_dict)
        '''
        for name, value in model.named_parameters():
            print(name)
            if ('last_linear' in name) or ('downsamp_modules' in name) or ('final_layer' in name) or ('incre_modules' in name) or ('stage4' in name): 
                value.requires_grad = True
            else:
                value.requires_grad = False
        '''
        '''
        for para in model.parameters():
            para.requires_grad = False

        for para in model.last_linear.parameters():
            para.requires_grad = True
        '''
	#model.load_state_dict(state_dict)
        print('hrnet pretrained is found')
        #model.last_linear = model.fc
        #del model.fc
    

    return model
    
class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, dropout=0.0):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception(pretrained='imagenet')
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'resnet50' or modelchoice == 'resnet101':
            if modelchoice == 'resnet50':
                self.model = torchvision.models.resnet50(pretrained=False)
            if modelchoice == 'resnet101':
                self.model = torchvision.models.resnet152(pretrained=True)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'mynet':
            self.model = return_mynet_xception()
            # Replace fc
            num_ftrs = self.model.last_linear.in_features
            if not dropout:
                self.model.last_linear = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.last_linear = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        elif modelchoice == 'hrnet':
            self.model = return_hrnet(True)
            # Replace fc

        else:
            raise Exception('Choose valid model, e.g. resnet50')
    """
    def set_trainable_up_to(self, boolean, layername="Conv2d_4a_3x3"):
        '''
        Freezes all layers below a specific layer and sets the following layers
        to true if boolean else only the fully connected final layer
        :param boolean:
        :param layername: depends on network, for inception e.g. Conv2d_4a_3x3
        :return:
        '''
        # Stage-1: freeze all the layers
        if layername is None:
            for i, param in self.model.named_parameters():
                param.requires_grad = True
                return
        else:
            for i, param in self.model.named_parameters():
                param.requires_grad = False
        if boolean:
            # Make all layers following the layername layer trainable
            ct = []
            found = False
            for name, child in self.model.named_children():
                if layername in ct:
                    found = True
                    for params in child.parameters():
                        params.requires_grad = True
                ct.append(name)
            if not found:
                raise Exception('Layer not found, cant finetune!'.format(
                    layername))
        else:
            if self.modelchoice == 'xception':
                # Make fc trainable
                for param in self.model.last_linear.parameters():
                    param.requires_grad = True

            else:
                # Make fc trainable
                for param in self.model.fc.parameters():
                    param.requires_grad = True
    """
    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes,
                    dropout=None):
    """
    :param modelname:
    :return: model, image size, pretraining<yes/no>, input_list
    """
    if modelname == 'xception':
        return TransferModel(modelchoice='xception',
                             num_out_classes=num_out_classes), 299, \
               True, ['image'], None
    elif modelname == 'resnet101':
        return TransferModel(modelchoice='resnet101', dropout=dropout,
                             num_out_classes=num_out_classes), \
               299, True, ['image'], None
    elif modelname == 'mynet':
        return TransferModel(modelchoice='mynet', num_out_classes=num_out_classes, dropout=dropout), \
                299, False, ['image'], None
    elif modelname == 'hrnet':
        return TransferModel(modelchoice='hrnet', num_out_classes=num_out_classes, dropout=dropout), \
                256, False, ['image'], None
    else:
        raise NotImplementedError(modelname)


if __name__ == '__main__':
    model, image_size, *_ = model_selection('hrnet', num_out_classes=2)
    print(model)
    model = model.cuda()
    x = torch.randn(1,3,299,299).cuda()
    y = model(x)
    print(y)
    """   
    from torchsummary import summary
    input_s = (3, image_size, image_size)
    print(summary(model, input_s))
    """
