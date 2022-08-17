# -*- coding = utf-8 -*-
# @Time : 2022/7/13 8:03
# @Author : 牧川
# @File : test_totally_train.py

import torchvision
import torch.nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from nn_CIFAR10 import MyModule

#初始化一个device
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print(f"当前的device类型：{device}")

# 准备数据集
train_data = torchvision.datasets.CIFAR10("datasets", train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)
test_data = torchvision.datasets.CIFAR10("datasets", train=False, transform=torchvision.transforms.ToTensor(),
                                         download=True)
# 数据集长度
train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集的长度：{test_data_size}\n测试数据集的长度：{test_data_size}")

# 准备dataloader：
train_dataloader = DataLoader(train_data, 64)
test_dataloader = DataLoader(test_data, 64)

# 搭建神经网络：
mymodle = MyModule()
mymodle.to(device)
# 损失函数：
loss_fn = torch.nn.CrossEntropyLoss()
loss_fn.to(device)
# 优化器：
learning_rate = 0.01
optim = torch.optim.SGD(mymodle.parameters(), lr=learning_rate)

# 声明一些计数变量
total_train_step = 0
total_test_step = 0
epoch = 20
# 添加tensorboard
writer = SummaryWriter("logs")

for i in range(epoch):
    print(f"第{i + 1}轮训练开始：")
    # 训练步骤:
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = mymodle(imgs)
        # print(outputs)
        result_loss = loss_fn(outputs, targets)  # 损失函数的计算
        optim.zero_grad()  # 梯度归零
        result_loss.backward()  # 梯度反向传播
        optim.step()  # 优化器
        total_train_step += 1

        if total_train_step % 100 == 0:
            print(f"训练次数{total_train_step},当前loss:{result_loss.item()}")
            writer.add_scalar("train_loss", result_loss.item(), total_train_step)

    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        # 每次训练结束后用测试集验证一次结果，no_grad即不对梯度造成影响
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)
            targets = targets.to(device)
            outputs = mymodle(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss

            #正确率计算：
            accuracy = (outputs.argmax(1) == targets).sum()
            """outputs.argmax(1)将每一行（原结构是对每一类的预测概率），改为每一行最大值的index，实际代表预测的类别"""
            """== targets即将预测结果与目标结果对比，得到true、false组成的数组"""
            """.sum()即将结果求和，实际代表命中的次数"""
            total_accuracy += accuracy
            """将本loader中数据的命中次数加至总和"""
    print(f"第{i + 1}轮训练结束，整体测试集上的loss:{total_test_loss}")
    print(f"第{i + 1}轮训练结束，整体测试集上的正确率:{total_accuracy/test_data_size}")
    writer.add_scalar("test_loss", loss.item(), total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

writer.close()

torch.save(mymodle.state_dict(), f"mymodle.pth")
print("训练已完成，模型已保存！")