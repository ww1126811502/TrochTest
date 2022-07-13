# -*- coding = utf-8 -*-
# @Time : 2022/7/10 17:51
# @Author : 牧川
# @File : test_all_01.py
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)
"""创建数据集，存放在datasets，为测试数据集，且转换为tensor"""

dataloader = DataLoader(dataset, 64)
"""用dataloader取数据集，设置每次打包64个"""


class TestModule(nn.Module):
    """构建神经网络类"""

    def __init__(self):
        super(TestModule, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=(3, 3), stride=(1, 1), padding=0)
        """卷积的各项参数"""

    def forward(self, x):
        x = self.conv1(x)
        """对输入进行卷积，并返回结果"""
        return x


module = TestModule()
"""实例化一个神经网络"""
print(module)

writer = SummaryWriter("logs")
"""实例化一个writer，用于观测训练过程"""

step = 0
for data in dataloader:
    imgs, targets = data
    """从dataloader中取到的imgs和targets均为数组"""
    output = module(imgs)
    """将每组imgs作为输入，放入神经网络得到输出"""
    writer.add_images("input", imgs, step)
    """输入为3通道，可以直接展示"""
    output = torch.reshape(output, (-1, 3, 30, 30))
    """输出为6通道，需要先转换一次,reshape的参数填-1代表自动"""
    writer.add_images("output", output, step)
    """输出为6通道，需要先转换一次"""

    step += 1
