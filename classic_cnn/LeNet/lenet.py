import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, Dropout, Activation, MaxPool2D, Flatten, Dense

np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255., x_test / 255.

# 经典CNN网络LeNet实现
class LeNet(Model):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5))  #
        self.a1 = Activation('sigmoid')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)  #

        self.c2 = Conv2D(filters=16, kernel_size=(5, 5))  #
        self.a2 = Activation('sigmoid')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)  #

        self.flatten = Flatten()
        self.f1 = Dense(120, activation='sigmoid')
        self.f2 = Dense(84, activation='sigmoid')
        self.f3 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.a1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.a2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y


model = LeNet()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_savepath = './checkpoint/LeNet.ckpt'
if os.path.exists(checkpoint_savepath + '.index'):
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_savepath)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_savepath,
                                                 save_weights_only=True,
                                                 save_best_only=True)

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), validation_freq=1,
                    callbacks=[cp_callback])

model.summary()

# # 把训练好的参数存入txt文本
# print(model.trainable_variables)
# file = open('./weights.txt', 'w')
# for v in model.trainable_variables:
#     file.write(str(v.name) + '\n')
#     file.write(str(v.shape) + '\n')
#     file.write(str(v.numpy()) + '\n')
# file.close()

# 显示训练集和验证集的acc和loss曲线
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='training acc')
plt.plot(val_acc, label='validation acc')
plt.title('Training and Validation acc')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='training loss')
plt.plot(val_loss, label='validation loss')
plt.title('Training and Validation loss')
plt.legend()
plt.show()

# Epoch 5/5
# 1563/1563 [==============================] - 14s 9ms/step - loss: 1.5309 - sparse_categorical_accuracy: 0.4427
# - val_loss: 1.4721 - val_sparse_categorical_accuracy: 0.4630





