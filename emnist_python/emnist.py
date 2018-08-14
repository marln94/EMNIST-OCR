import tensorflow as tf
import tensorflowjs as tfjs
import pandas as pd
from sklearn.model_selection import train_test_split

raw_data = pd.read_csv("data/emnist-balanced-train.csv")

train, test = train_test_split(raw_data, test_size=0.1)

x_train = train.values[:,1:]
print(train.values)
print(x_train)
y_train = train.values[:,0]
print(y_train)

x_test = test.values[:,1:]
y_test = test.values[:,0]

# x_train = x_train.reshape(-1, 28 * 28) / 255.0
# x_test = x_test.reshape(-1, 28 * 28) / 255.0
# x_train = x_train[:25000]
# x_test = x_test[:25000]
# y_train = y_train[:25000]
# y_test = y_test[:25000]


x_train = x_train.reshape(x_train.shape[0], 28, 28, 1) / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1) / 255.0
input_shape = (28, 28, 1)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'validation samples')


## Arquitectura a implementar
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, 3, strides=1, activation=tf.nn.relu, input_shape=input_shape),
  tf.keras.layers.MaxPooling2D((2, 2), (2, 2)),
  tf.keras.layers.Conv2D(32, 2, strides=1, activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D((2, 2), (2, 2)),
  tf.keras.layers.Conv2D(64, 3, strides=1, activation=tf.nn.relu),
  tf.keras.layers.MaxPooling2D((2, 2), (2, 2)),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation=tf.nn.relu),
  # tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(47, activation=tf.nn.softmax)
])

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Dense(1024, activation=tf.nn.relu, input_shape=(784,)),
#     tf.keras.layers.Dropout(0.5),
#     tf.keras.layers.Dense(47, activation=tf.nn.softmax)
#   ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',
#               optimizer=tf.keras.optimizers.SGD(lr=0.15),
#               metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5, batch_size=5)
print("evaluando")
model.evaluate(x_test, y_test)

tfjs.converters.save_keras_model(model, "model/")