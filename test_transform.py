# -*- coding = utf-8 -*-
# @Time : 2022/7/8 20:56
# @Author : 牧川
# @File : test_transform.py
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# tensor数据类型？

img_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(img_path)

writer = SummaryWriter("logs")
# ToTensor的使用：
tensor_trans = transforms.ToTensor()  # 实例化一个ToTensor类
tensor_img = tensor_trans(img)  # 将图片转化为tensor格式
writer.add_image("tensor_img", tensor_img)

# Normalize:
print(tensor_img[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # 实例化一个Normalize类，规定了如何处理tensor
img_norm = trans_norm(tensor_img)
print(tensor_img[0][0][0])
writer.add_image("norm_img", img_norm)

# Resize：注意输入为PIL_img
trans_resize = transforms.Resize((512, 512))  # 初始化时设置要输出的大小
img_resize = trans_resize(img)  # 输入和输出均为PIL格式下的img

# Compose: 可以合并多个transform
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])  # 入参为一个transform数组
img_resize_2 = trans_compose(img)
writer.add_image("Resize", img_resize_2)

writer.close()
