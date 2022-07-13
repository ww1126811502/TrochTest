# -*- coding = utf-8 -*-
# @Time : 2022/7/9 15:22
# @Author : 牧川
# @File : test_nn.module.py
import torch
from torch import nn


#神经网络类的基础声明+用法
class DemoModule(nn.Module):
    """神经网络对象需要重写初始化函数+forward函数"""
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input+1
        return


MyModule = DemoModule()
x = torch.tensor(1.0)
output = MyModule(x)