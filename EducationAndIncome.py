import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('./data.csv')
# print(data)
# plt.scatter(data.Education, data.Income)
# plt.show()

x = data.Education
y = data.Income
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(1, input_shape=(1, )))
# model.summary()
model.compile(
    optimizer='adam',
    loss='mse'
)
history = model.fit(x, y, epochs=5000)
result = model.predict(pd.Series([20]))
print(result)

