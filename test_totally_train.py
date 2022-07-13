# -*- coding = utf-8 -*-
# @Time : 2022/7/13 8:03
# @Author : 牧川
# @File : test_totally_train.py

import torchvision
import torch.nn
from torch.utils.data import DataLoader
from nn_CIFAR10 import MyModule

#准备数据集
train_data = torchvision.datasets.CIFAR10("datasets", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
#数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度：{test_data_size}\n测试数据集的长度：{test_data_size}")

#准备dataloader：
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

#搭建神经网络：
mymodle = MyModule()
#损失函数：
loss_fn = torch.nn.CrossEntropyLoss()
#优化器：
learning_rate = 0.01
optim = torch.optim.SGD(mymodle.parameters(), lr=learning_rate)

#声明一些计数变量
total_train_step = 0
total_test_step = 0
epoch = 10

for i in range(epoch):
    print(f"第{i+1}轮训练开始：")
    #训练步骤:
    for data in train_data:
        imgs, targets = data
        outputs = mymodle(imgs)
        # print(outputs)
        result_loss = mymodle(outputs, targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()
        total_train_step += 1
    print(f"第{i + 1}轮训练结束，")