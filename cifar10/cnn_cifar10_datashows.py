import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 看下数据长啥样，简单了解数据
plt.imshow(x_train[0])  # 查看图片的方式
plt.show()

# 数据全貌
print("x_train[0]: \n", x_train[0])
print("y_train[0]: \n", y_train[0])
print("x_train.shape: \n", x_train.shape)
print("y_train.shape: \n", y_train.shape)
print("x_test.shape: \n", x_test.shape)
print("y_test.shape: \n", y_test.shape)
