import tensorflow as tf
import tensorflowjs as tfjs
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

y_train = y_train[:1000]
y_test = y_test[:1000]

#x_train, x_test = x_train / 255.0, x_test / 255.0

# x_train = x_train[:1000].reshape(x_train.shape[0], 28, 28, 1) / 255.0
# x_test = x_test[:1000].reshape(x_test.shape[0], 28, 28, 1) / 255.0

x_train = x_train[:1000].reshape(-1, 28 * 28) / 255.0
x_test = x_test[:1000].reshape(-1, 28 * 28) / 255.0

print(x_test.shape)

### Arquitectura a implementar
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Conv2D(8, 5, strides=1, activation=tf.nn.relu, input_shape=(784,), kernel_initializer=tf.initializers.variance_scaling),
#   tf.keras.layers.MaxPooling2D((2, 2), (2, 2)),
#   tf.keras.layers.Conv2D(5, 16, strides=1, activation=tf.nn.relu, kernel_initializer=tf.initializers.variance_scaling),
#   tf.keras.layers.MaxPooling2D((2, 2), (2, 2)),
#   tf.keras.layers.Flatten(),
#   tf.keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer=tf.initializers.variance_scaling)
# ])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
  ])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
print("evaluando")
model.evaluate(x_test, y_test)

tfjs.converters.save_keras_model(model, "model/")