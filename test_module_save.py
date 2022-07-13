# -*- coding = utf-8 -*-
# @Time : 2022/7/13 7:35
# @Author : 牧川
# @File : test_module_save.py
import torch
import torchvision.models

vgg16 = torchvision.models.vgg16(pretrained=False)

#保存方式1，保存的是模型结构+模型参数
torch.save(vgg16, "modules/vgg16_test.pth")

#加载方式1：
model = torch.load("modules/vgg16_test.pth")

#保存方式2：仅保存模型参数，官方推荐
torch.save(vgg16.state_dict(), "modules/vgg16_test2.pth")
#对应的加载方式：
vgg16_new = torchvision.models.vgg16(pretrained=False)
vgg16_new.load_state_dict(torch.load("modules/vgg16_test2.pth"))
