# -*- coding = utf-8 -*-
# @Time : 2022/7/9 9:35
# @Author : 牧川
# @File : test_dataloader.py
import torchvision
from torch.utils.data import DataLoader

#准备一个数据集
test_set = torchvision.datasets.CIFAR10("./datasets", train=False, transform=torchvision.transforms.ToTensor(), download=True)
#对测试数据集，每次取出4个，主进程且不舍弃余数数据
test_loader = DataLoader(dataset=test_set, batch_size=4, num_workers=0, drop_last=False)

for data in test_loader:
    imgs, targets = data
    print(f"imgs.shape：{imgs.shape},targets:{targets}")
