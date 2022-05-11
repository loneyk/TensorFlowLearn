import tensorflow as tf
import numpy as np

# tf.where用法
a = tf.constant([1, 2, 3, 1, 1])
b = tf.constant([0, 1, 3, 4, 5])
c = tf.where(tf.greater(a, b), a, b)  # 若a>b,返回a对应位置的元素，否则返回b对应位置的元素
print("c:", c)

# np.random.RandomState.rand(维度)
rdm = np.random.RandomState(seed=1)  # seed=常数表示生成的随机数相同
a = rdm.rand()  # 返回一个随机标量
b = rdm.rand(2, 3)  # 返回维度为2行3列的随机数矩阵
print("a:", a)
print("b:", b)

# np.vstack将两个数组按照垂直方向叠加
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
c = np.vstack((a, b))
print("c:\n", c)

# 网格坐标点
# np.mgrid[起始值:结束值:步长,''',]
# x.ravel()把x变成一维数组，垂直方向拉直
# np.c_[]使返回的间隔数值点配对
x, y = np.mgrid[1:3:1, 2:4:0.5]  # 分别对应x轴和y轴的数据，一般这三个函数配套使用
grid = np.c_[x.ravel(), y.ravel()]
print("x:", x)
print("y:", y)
print("grid:\n", grid)
