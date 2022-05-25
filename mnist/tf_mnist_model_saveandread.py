# 1. 读取模型
# 2. 把训练好的参数写入文本txt
# 3. 画出acc & loss曲线

import tensorflow as tf
import os
from matplotlib import pyplot as plt
import numpy as np
np.set_printoptions(threshold=np.inf)  # 参数全部打印，不使用省略号

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 增加模型读取操作
checkpoint_save_path = "./checkpoint/mnist.ckpt"
if os.path.exists(checkpoint_save_path + '.index'):    # 如果模型存在，直接load读取出来，那么接下来的训练过程就是在这个基础上进行的
    print('-------------load the model-----------------')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
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
