# -*- coding = utf-8 -*-
# @Time : 2022/7/10 21:22
# @Author : 牧川
# @File : test_nn.MaxPool2d.py
import torch
import torchvision.datasets
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

datasets = torchvision.datasets.CIFAR10("datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                        download=True)
dataloader = DataLoader(datasets, 64)


class TestModule(nn.Module):

    def __init__(self):
        super(TestModule, self).__init__()
        self.maxPool = MaxPool2d(kernel_size=3, ceil_mode=True)
        """初始化一个maxPool池，池化核尺寸为3x3，开启ceil模式"""

    def forward(self, input):
        output = self.maxPool(input)
        return output


testmodule = TestModule()
step = 0
writer = SummaryWriter("logs")
for data in dataloader:
    imgs, targets = data
    writer.add_images("input2", imgs, step)
    """展示原始图片"""
    output = testmodule(imgs)
    writer.add_images("output2", output, step)
    """展示处理后的图片"""
    step += 1

writer.close()
