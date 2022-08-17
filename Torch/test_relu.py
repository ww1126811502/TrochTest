# -*- coding = utf-8 -*-
# @Time : 2022/7/11 22:02
# @Author : 牧川
# @File : test_relu.py
import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
input = torch.reshape(input, (-1, 1, 2, 2))


class MyModule(nn.Module):
    def __init__(self):
        super(MyModule, self).__init__()
        self.relu = ReLU()

    def forward(self, input):
        return self.relu(input)


mymodule = MyModule()
output = mymodule(input)
print(output)