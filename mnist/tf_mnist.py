from abc import ABC
import tensorflow as tf
from PIL import Image
import numpy as np
import os
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Model

# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0  # 归一化，使值都在0-1之间，方便神经网络计算、

train_path = './mnist_image_label/mnist_train_jpg_60000/'
train_txt = './mnist_image_label/mnist_train_jpg_60000.txt'
x_train_savepath = './mnist_image_label/mnist_x_train.npy'
y_train_savepath = './mnist_image_label/mnist_y_train.npy'

test_path = './mnist_image_label/mnist_test_jpg_10000/'
test_txt = './mnist_image_label/mnist_test_jpg_10000.txt'
x_test_savepath = './mnist_image_label/mnist_x_test.npy'
y_test_savepath = './mnist_image_label/mnist_y_test.npy'


def generateds(path, txt):
    with open(txt, mode='r') as f:
        contents = f.readlines()
    x, y_ = [], []
    for content in contents:
        value = content.split()
        img_path = path + value[0]

        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                img = np.array(img.convert('L'))
                img = img / 255.
                x.append(img)
                y_.append(value[1])
                # print('loading: ' + content)
        # else:
        #     print('Path: ' + img_path + ' not found.')

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)

    print("x shape: " + str(x.shape[0]))  # 打印共有多少张训练/测试图片
    print("y_ shape: " + str(y_.shape[0]))
    return x, y_


if (os.path.exists(x_train_savepath) and os.path.exists(y_train_savepath)
        and os.path.exists(x_test_savepath) and os.path.exists(y_test_savepath)):
    print('-------------Load Datasets-----------------')
    x_train_save = np.load(x_train_savepath)
    y_train = np.load(y_train_savepath)
    x_test_save = np.load(x_test_savepath)
    y_test = np.load(y_test_savepath)
    x_train = np.reshape(x_train_save, (len(x_train_save), 28, 28))
    x_test = np.reshape(x_test_save, (len(x_test_save), 28, 28))
else:
    print('-------------Generate DataSets-----------------')
    x_train, y_train = generateds(train_path, train_txt)
    x_test, y_test = generateds(test_path, test_txt)

    print('-------------Save Datasets-----------------')
    x_train_save = np.reshape(x_train, (len(x_train), -1))
    x_test_save = np.reshape(x_test, (len(x_test), -1))
    np.save(x_train_savepath, x_train_save)
    np.save(y_train_savepath, y_train)
    np.save(x_test_savepath, x_test_save)
    np.save(y_test_savepath, y_test)


# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])
class MnistModel(Model, ABC):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        y = self.d2(x)
        return y


model = MnistModel()

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

model.fit(x_train, y_train, batch_size=32, epochs=8, validation_data=(x_test, y_test), validation_freq=2)
model.summary()
