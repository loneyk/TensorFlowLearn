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

class BaseLine(Model):
    def __init__(self):
        super(BaseLine, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), padding='same')  #
        self.b1 = BatchNormalization()
        self.a1 = Activation('relu')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2, padding='same')  #
        self.d1 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(128, activation='relu')
        self.d2 = Dropout(0.2)
        self.f2 = Dense(10, activation='softmax')

    def call(self, x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)
        x = self.flatten(x)
        x = self.f1(x)
        x = self.d2(x)
        y = self.f2(x)
        return y


model = BaseLine()
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

checkpoint_savepath = './checkpoint/Baseline.ckpt'
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
# 1563/1563 [==============================] - 26s 17ms/step - loss: 1.2471 - sparse_categorical_accuracy: 0.5541
# - val_loss: 1.2318 - val_sparse_categorical_accuracy: 0.5703





