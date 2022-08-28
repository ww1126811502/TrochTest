# -*- coding = utf-8 -*-
# @Time : 2022/8/28 22:27
# @Author : 牧川
# @File : test.py
import matplotlib.pyplot as plt

# 第1步：定义x和y坐标轴上的点   x坐标轴上点的数值
x = [1, 2, 3, 4]
# y坐标轴上点的数值
y = [1, 4, 9, 16]
# 第2步：使用plot绘制线条第1个参数是x的坐标值，第2个参数是y的坐标值
plt.plot(x, y, color='b')
# 第3步：显示图形
plt.show()
