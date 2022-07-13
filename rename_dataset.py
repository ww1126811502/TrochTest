# -*- coding = utf-8 -*-
# @Time : 2022/7/6 22:54
# @Author : 牧川
# @File : rename_dataset.py
import os

root_dir = "dataset\\train"
target_dir = "bees_image"
image_path = os.listdir(os.path.join(root_dir, target_dir))
label = target_dir.split("_")[0]
out_dir = "bees_label"

for i in image_path:
    file_name = i.split(".jpg")[0]
    with open(os.path.join(root_dir, out_dir, f'{file_name}.txt'), 'w') as f:
        f.write(label)