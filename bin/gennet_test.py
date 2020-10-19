import tensorflow as tf
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
#
# input = tf.keras.Input(shape=(100,), dtype='int32', name='input')
# x = tf.keras.layers.Embedding(
#     output_dim=512, input_dim=10000, input_length=100)(input)
# x = tf.keras.layers.LSTM(32)(x)
# x = tf.keras.layers.Dense(64, activation='relu')(input)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# x = tf.keras.layers.Dense(64, activation='relu')(x)
# output = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(x)
# model = tf.keras.Model(inputs=[input], outputs=[output])
# dot_img_file = '/tmp/model_1.png'
# tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
#
#
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(1288701, 1)),
  tf.keras.layers.Conv1D(filters=60, kernel_size=60, strides=60),
  tf.keras.layers.Dense(21390, activation='relu'),
  tf.keras.layers.Dense(1, activation='softmax')
])
# this is a logistic regression in Keras
# x = tf.keras.layers.Input(shape=(1288701,))
# x = tf.keras.layers.Dense(21390, activation='relu')(x)
# y = tf.keras.layers.Dense(1, activation='sigmoid')(x)
# model = tf.keras.Model(x, y)
model.summary()

tf.keras.utils.plot_model(
    model, to_file='model.png', show_shapes=True, show_layer_names=True,
    rankdir='TB',
)

exit()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)