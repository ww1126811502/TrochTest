# -*- coding = utf-8 -*-
# @Time : 2022/7/9 15:46
# @Author : 牧川
# @File : test_nn.conv.py
# 卷积的基础用法，主要是2D卷积，所以对二维数组进行计算
import torch
import torch.nn.functional as F

# 模拟输入图像
testInput = torch.tensor([[1, 2, 0, 3, 1],
                          [0, 1, 2, 3, 1],
                          [1, 2, 1, 0, 0],
                          [5, 2, 3, 1, 1],
                          [2, 1, 0, 1, 1]])
# 卷积核
kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])
# conv2需要入参带有channel
# 自定义的tensor变量只有高和宽，没有channel
# 需要用reshape变换尺寸（即shape）
testInput = torch.reshape(testInput, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

out = F.conv2d(testInput, kernel, stride=1)
print(out)

out2 = F.conv2d(testInput, kernel, stride=1, padding=1)
print(out2)