# -*- coding = utf-8 -*-
# @Time : 2022/7/12 23:17
# @Author : 牧川
# @File : test_module_use.py
import torchvision.datasets
from torch import nn

train_data = torchvision.datasets.ImageNet("./data_image_net", split="train",
                                           transform=torchvision.transforms.ToTensor())
vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)

vgg16_true.add_module("name", nn.Linear(1000, 10))