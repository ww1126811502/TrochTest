# -*- coding = utf-8 -*-
# @Time : 2022/7/15 22:51
# @Author : 牧川
# @File : test_use.py
import torch
import torchvision.transforms as trans
from PIL import Image
from nn_CIFAR10 import MyModule

img_path = 'imgs/tens_2.png'
img = Image.open(img_path)
print(img.format)
#png是四通道，需要转通道
if img.format == 'PNG':
    img = img.convert("RGB")
#print(img)

transform2 = trans.Compose([trans.Resize((32, 32)),
                           trans.ToTensor()])
img = transform2(img)
#print(f"img.shape={img.shape}")
print(f"图片{img_path}读取并转换成功")

module = MyModule()
module.load_state_dict(torch.load("mymodle.pth"))
print("模型加载成功")

img = torch.reshape(img, (1, 3, 32, 32))
module.eval()   #转换至测试状态
with torch.no_grad():
    outputs = module(img)
#转换一下测试结果
print(f"预测的结果是：{module.type[outputs.argmax(1)]}")
