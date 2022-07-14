# -*- coding = utf-8 -*-
# @Time : 2022/7/12 1:00
# @Author : 牧川
# @File : nn_CIFAR10.py
import torch
import torchvision
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def main():
    print("加载成功")


def moduletest():
    dataset = torchvision.datasets.CIFAR10("datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                           download=True)
    """创建数据集，存放在datasets，为测试数据集，且转换为tensor"""

    dataloader = DataLoader(dataset, 64)
    """用dataloader取数据集，设置每次打包64个"""

    module = MyModule()
    # 测试部分：
    # input = torch.ones((64, 3, 32, 32))
    # """torch.ones，模拟一个与输入格式相同的tensor，便于进行测试和检查"""
    # output = module(input)
    # print(output.shape)
    #
    # writer = SummaryWriter("logs")
    # writer.add_graph(module, input)
    # """利用graph展示，可以清晰看到数据的每一步转化及其参数"""
    #
    #
    # writer.close()

    loss = nn.CrossEntropyLoss()
    step = 0
    for data in dataloader:
        imgs, targets = data
        """从dataloader中取到的imgs和targets均为数组"""
        outputs = module(imgs)
        # print(outputs)
        result_loss = loss(outputs, targets)
        print(result_loss)
        result_loss.backward()
        print(result_loss)


class MyModule(nn.Module):
    """搭建神经网络，用于处理CIFAR10"""

    def __init__(self):
        super(MyModule, self).__init__()
        self.model1 = Sequential(
            Conv2d(3, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, 1, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


if __name__ == '__main__':
    module = MyModule()
    input = torch.ones((64, 3, 32, 32))
    output = module(input)
    print(output.shape)
