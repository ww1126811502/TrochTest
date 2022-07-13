# -*- coding = utf-8 -*-
# @Time : 2022/7/6 21:39
# @Author : 牧川
# @File : read_data.py
from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    """自定义数据集类，继承Dataset"""

    def __init__(self, root_dir, label_dir):
        """
        数据集初始化函数
        :param root_dir: 数据集的根目录
        :param label_dir: 数据集名称
        """
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        """数据集的真实目录"""
        self.img_path = os.listdir(self.path)
        """包含数据集下所有文件名的数组"""

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        """当前要获取的对象名称"""
        img_item_path = os.path.join(self.path, img_name)
        """当前要获取对象的实际路径"""
        img = Image.open(img_item_path)
        """图片对象"""
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "dataset/train"
ants_label_dir = "ants_image"
bees_label_dir = "bees_image"
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
print(ants_dataset[1])