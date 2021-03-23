import tensorflow as tf
from sklearn import datasets
import numpy as np

# 导入训练集 以及训练集的标签
x_train = datasets.load_iris().data
y_train = datasets.load_iris().target
# 实现对数据集的乱序
# seed( ) 用于指定随机数生成时所用算法开始的整数值。
# 1.如果使用相同的seed( )值，则每次生成的随即数都相同；
np.random.seed(116)
np.random.shuffle(x_train) #打乱数据集
np.random.seed(116)
np.random.shuffle(y_train)
tf.random.set_seed(116)

# 搭建模型 神经元的个数 选用的激活函数 选用的正则化方法
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2())
])

# compile选用训练的配置方法 选择SGD优化器 学习率设置为0.1
# loss损失函数设置为SparseCategoricalCrossentropy
# from_logits=False 寻问是否是原始输出 由于前面使用了softmax激活函数的概率分布 所以是false
# metrics评测指标 如果选择accuracy 则对应情况为y_和y都是数值形式 y_=[1] y=[1]
# sparse_categorical_accuracy对应y_和y都是以独热编码或者概率分布形式给出
# y_=[1]是数值形式  但是模型输出为y=[0.256,0.695,0.048]
model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# 测试集选择从训练集中划分 一次喂入多少组数据 batch_size 循环多少次
# validation_split告知从训练集中选择20%的数据作为测试集
# validation_freq 每迭代20次训练集要在测试集中验证一次准确率
model.fit(x_train, y_train, batch_size=32, epochs=500, validation_split=0.2, validation_freq=20)

# 用summary打印出网络结构和参数统计
model.summary()

