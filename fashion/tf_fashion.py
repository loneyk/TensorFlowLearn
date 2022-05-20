import tensorflow as tf
# from matplotlib import pyplot as plt

# 数据处理 构建特征工程 这一步其实很重要
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train/255.0, x_test/255.0  # 归一化处理

# 搭建网络，首先是根据经验搭建，然后再调整
# 96 -> 32 ->10 :
# val_loss: 0.7641 - val_sparse_categorical_accuracy: 0.8831
# 128 -> 10:
# val_loss: 0.3398 - val_sparse_categorical_accuracy: 0.8818
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 网络参数的优化， 选取何种优化器，以及损失函数和准确度的计算等
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics='sparse_categorical_accuracy')

# 训练集输入与标签，batch size 一次喂入多少组数据，epoch循环多少次， validation_data 测试集，validation_freq训练几轮做一次验证
model.fit(x_train, y_train, batch_size=32, epochs=8, validation_freq=2, validation_data=(x_test, y_test))

# 神经网络总结
model.summary()

