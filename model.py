import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        self.input_num_capsule = input_shape[1]
        self.input_dim_capsule = input_shape[2]

        self.W = self.add_weight(shape=[self.input_num_capsule, self.num_capsule, self.input_dim_capsule, self.dim_capsule],
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 name='W')

    def call(self, inputs, training=None):
        inputs_expand = tf.expand_dims(inputs, 2)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsule, 1])
        inputs_hat = tf.map_fn(lambda x: tf.matmul(x, self.W), elems=inputs_tiled)
        b = tf.zeros(shape=[tf.shape(inputs_hat)[0], self.input_num_capsule, self.num_capsule, 1])

        assert self.routings > 0
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            outputs = squash(tf.reduce_sum(c * inputs_hat, axis=1, keepdims=True))
            if i < self.routings - 1:
                b += tf.reduce_sum(inputs_hat * outputs, axis=-1, keepdims=True)
        return tf.squeeze(outputs, axis=1)

def Length(name):
    return layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1)), name=name)

def Mask():
    return layers.Lambda(lambda x: x[0] * tf.expand_dims(x[1], -1))

def squash(vectors, axis=-1):
    s_squared_norm = tf.reduce_sum(tf.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / tf.sqrt(s_squared_norm + tf.keras.backend.epsilon())
    return scale * vectors

def margin_loss(y_true, y_pred):
    L = y_true * tf.square(tf.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * tf.square(tf.maximum(0., y_pred - 0.1))
    return tf.reduce_mean(tf.reduce_sum(L, 1))

def CapsNet(input_shape, n_class, routings):
    x = layers.Input(shape=input_shape)
    conv1 = layers.Conv2D(filters=256, kernel_size=9, strides=1, padding='valid', activation='relu', name='conv1')(x)
    primarycaps = PrimaryCap(conv1, dim_capsule=8, n_channels=32, kernel_size=9, strides=2, padding='valid')
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)
    out_caps = Length(name='out_caps')(digitcaps)
    y = layers.Input(shape=(n_class,))
    masked_by_y = Mask()([digitcaps, y])
    masked = Mask()(digitcaps)
    decoder = models.Sequential(name='decoder')
    decoder.add(layers.Dense(512, activation='relu', input_dim=16*n_class))
    decoder.add(layers.Dense(1024, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(target_shape=input_shape, name='out_recon'))
    train_model = models.Model([x, y], [out_caps, decoder(masked_by_y)])
    eval_model = models.Model(x, [out_caps, decoder(masked)])
    noise = layers.Input(shape=(n_class, 16))
    noised_digitcaps = layers.Add()([digitcaps, noise])
    masked_noised_y = Mask()([noised_digitcaps, y])
    manipulate_model = models.Model([x, y, noise], decoder(masked_noised_y))
    return train_model, eval_model, manipulate_model

def get_model():
    model, _, _ = CapsNet(input_shape=(28, 28, 1), n_class=10, routings=3)
    model.compile(optimizer=optimizers.Adam(lr=0.001),
                  loss=[margin_loss, 'mse'],
                  loss_weights=[1., 0.392],
                  metrics={'out_caps': 'accuracy'})
    return model

def train_model(model, train_data, train_labels):
    model.fit([train_data, train_labels], [train_labels, train_data],
              batch_size=100, epochs=50, validation_split=0.2)

def evaluate_model(model, test_data, test_labels):
    return model.evaluate([test_data, test_labels], [test_labels, test_data])
