# -*- coding = utf-8 -*-
# @Time : 2022/7/12 22:49
# @Author : 牧川
# @File : test_optim.py

import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
"""创建数据集，存放在datasets，为测试数据集，且转换为tensor"""

dataloader = DataLoader(dataset, 64)
"""用dataloader取数据集，设置每次打包64个"""


class MyModule(nn.Module):
    """搭建神经网络，用于处理CIFAR10"""

    def __init__(self):
        super(MyModule, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        return self.model1(x)


module = MyModule()
optim = torch.optim.SGD(module.parameters(), lr=0.01)
"""初始化优化器，params参数可以直接用网络的parameters()方法返回"""
loss = nn.CrossEntropyLoss()
step = 0
for epoch in range(20):
    #设定进行20轮学习
    running_loss = 0.0
    for data in dataloader:
        #一轮学习：
        imgs, targets = data
        outputs = module(imgs)
        #print(outputs)
        result_loss = loss(outputs, targets)
        optim.zero_grad()
        """梯度归零"""
        result_loss.backward()
        """反向传播，更新梯度"""
        optim.step()
        """参数调优"""
        running_loss += result_loss
        """验证整体loss"""
    print(f'在第{epoch}轮中，误差和为：{running_loss}')