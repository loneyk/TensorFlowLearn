# 自制数据集
# 60000张训练图片和10000张测试图片，以及相对应的标签txt文件
# 通过以下的方式可以把最原始的数据，转换成tensorflow等神经网络认识的数据

from PIL import Image
import numpy as np
import os

train_path = './mnist_image_label/mnist_train_jpg_60000/'
train_txt = './mnist_image_label/mnist_train_jpg_60000.txt'
x_train_savepath = './mnist_image_label/mnist_x_train.npy'
y_train_savepath = './mnist_image_label/mnist_y_train.npy'

test_path = 'mnist/mnist_image_label/mnist_test_jpg_10000/'
test_txt = 'mnist/mnist_image_label/mnist_test_jpg_10000.txt'
x_test_savepath = './mnist_image_label/mnist_x_test.npy'
y_test_savepath = './mnist_image_label/mnist_y_test.npy'


def generateds(path, txt):
    with open(txt, mode='r') as f:
        contents = f.readlines()
    x, y_ = [], []
    for content in contents:
        value = content.split()
        img_path = path + value[0]

        with Image.open(img_path) as img:
            img = np.array(img.convert('L'))
            img = img / 255.
            x.append(img)
            y_.append(value[1])
            print('loading: ' + content)

    x = np.array(x)
    y_ = np.array(y_)
    y_ = y_.astype(np.int64)
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












