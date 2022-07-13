# -*- coding = utf-8 -*-
# @Time : 2022/7/13 8:03
# @Author : 牧川
# @File : test_totally_train.py

import torchvision
from torch.utils.data import DataLoader

#准备数据集
train_data = torchvision.datasets.CIFAR10("datasets", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
#数据集长度
test_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度：{test_data_size}\n测试数据集的长度：{test_data_size}")

#准备dataloader：
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

#搭建神经网络：
