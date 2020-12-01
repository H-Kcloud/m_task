# -*- coding: utf-8 -*-

import math
import numpy as np

'''
Warmup是在ResNet论文中提到的一种学习率预热的方法，它在训练开始的时候先选择使用一个较小的学习率，
训练了一些epoches或者steps(比如4个epoches,10000steps),再修改为预先设置的学习率来进行训练。
'''
def lr_warmup(lr_list, lr_init, warmup_end_epoch=5):
    lr_list[:warmup_end_epoch] = list(np.linspace(0, lr_init, warmup_end_epoch))
    return lr_list


def lr_scheduler(lr_init, num_epochs, warmup_end_epoch=5, mode='cosine'):
    """

    :param lr_init：initial learning rate
    :param num_epochs: number of epochs
    :param warmup_end_epoch: number of warm up epochs
    :param mode: {cosine}
                  cosine:
                        lr_t = 0.5 * lr_0 * (1 + cos(t * pi / T)) in t'th epoch of T epochs
    :return:
    """
    lr_list = [lr_init] * num_epochs

    print('*** learning rate warms up for {} epochs'.format(warmup_end_epoch))
    lr_list = lr_warmup(lr_list, lr_init, warmup_end_epoch)

    print('*** learning rate decays in {} mode'.format(mode))
    if mode == 'cosine':
        # 按cosine模式衰减并保证为正
        for t in range(warmup_end_epoch, num_epochs):
            cs = 1 + math.cos((t - warmup_end_epoch + 1) * math.pi / num_epochs)
            lr_list[t] = 0.5 * lr_init * cs
    else:
        raise AssertionError('{} mode is not implemented'.format(mode))
    return lr_list


if __name__ == '__main__':
    print('===> Test warm up')
    learning_rate_list = [1] * 20
    learning_rate_init_value = 0.01
    print(learning_rate_list)
    learning_rate_list = lr_warmup(learning_rate_list, learning_rate_init_value, 5)
    print(learning_rate_list)#[0.0, 0.0025, 0.005, 0.0075, 0.01, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    print('===> Test lr scheduler - cosine mode')
    learning_rate_list_scheduled = lr_scheduler(0.01, 100, 5)
    print(learning_rate_list_scheduled)
