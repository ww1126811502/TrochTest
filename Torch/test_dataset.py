# -*- coding = utf-8 -*-
# @Time : 2022/7/9 8:37
# @Author : 牧川
# @File : test_dataset.py
import torchvision
from torchvision import transforms as trans
from torch.utils.tensorboard import SummaryWriter

datasets_trans = trans.Compose([
    trans.ToTensor()
])
train_set = torchvision.datasets.CIFAR10("./datasets", train=True, transform=datasets_trans, download=True)
test_set = torchvision.datasets.CIFAR10("./datasets", train=False, transform=datasets_trans, download=True)
#返回值为图片+target，如果转化过，则是转化后的对象+target

writer = SummaryWriter("p10")
for img_tensor, target in train_set:
    writer.add_image("train_set", img_tensor, target)

writer.close()
