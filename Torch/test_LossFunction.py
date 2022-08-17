# -*- coding = utf-8 -*-
# @Time : 2022/7/12 21:54
# @Author : 牧川
# @File : test_LossFunction.py
import torch
from torch import nn
from torch.nn import L1Loss

inputs = torch.tensor([1, 3, 5], dtype=torch.float32)
outputs = torch.tensor([1, 3, 8], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
outputs = torch.reshape(outputs, (1, 1, 1, 3))

loss = L1Loss(reduction='sum')
print(loss(inputs, outputs))

#交叉熵：
x = torch.tensor([0.1, 0.2, 0.3])
x = torch.reshape(x, (1, 3))
"""x模拟了对三个类别的命中率分别为0.1,0.2,0.3"""
y = torch.tensor([1])
"""y代表要命中的是第[1]个类(顺序上的第二个)"""
loss_cross = nn.CrossEntropyLoss()

print(loss_cross(x, y))
"""-0.2+ln(exp(0.1)+exp(0.2)+exp(0.3))"""