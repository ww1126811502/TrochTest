# -*- coding = utf-8 -*-
# @Time : 2022/7/6 23:07
# @Author : 牧川
# @File : test_tb.py
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np

write = SummaryWriter("logs")
"""实例化一个SW对象，将结果存放在logs路径"""


#常用的两个方法：
#write.add_image()
#write.add_scalar()

#write.add_scalar()的用法：
# for i in range(100):
#     write.add_scalar("y=x", i, i)

#write.add_image()的用法：
image_path = "dataset/train/ants_image/0013035.jpg"
img = Image.open(image_path)
#需要一个numpy转化后的数组
img_array = np.array(img)
#对数组的格式shape有要求，可以先检查数组格式
print(img_array.shape)  #(512, 768, 3)代表(H, W, C)
write.add_image("test", img_array, 1, dataformats='HWC')
for i in range(100):
    write.add_scalar("y=2x", 3*i, i)

write.close()

