# 通过训练好的神经网络模型，导入模型参数，实现给图识物的基本功能
import tensorflow as tf
import numpy as np
from PIL import Image

model_save_path = './checkpoint/mnist.ckpt'

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.load_weights(model_save_path)

preNum = int(input("Please input the number of pred pictures: "))
for i in range(preNum):
    image_path = input("input the image path: ")
    img = Image.open(image_path)
    img = img.resize((28, 28), Image.ANTIALIAS)  # 以后训练数据的时候，一定要考虑好数据的维度，以及神经网络输入所要求的维度，格式是否一致
    img_arr = np.array(img.convert('L'))

    # 数据的预处理
    # 第一种方法：
    # img_arr = 255 - img_arr  # 图象值黑白翻转，因为输入要求是黑底白字，我们输入的图片是白底黑字

    # 第二种方法：
    for i in range(28):
        for j in range(28):
            if img_arr[i][j] < 200:
                img_arr[i][j] = 255
            else:
                img_arr[i][j] = 0

    img_arr = img_arr / 255.
    x_predict = img_arr[tf.newaxis, ...]  # 由于训练的时候都是由batch送入神经网络，所以这里要加上一个新的维度
    result = model.predict(x_predict)  # 返回值是一个tensor，代表所有类型的概率值

    pred = tf.argmax(result, axis=1)
    print('result: ', result)
    print('\n')
    tf.print(pred)














