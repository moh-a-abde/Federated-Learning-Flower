# model.py
from keras import layers, models, optimizers
from keras import backend as K
from keras.utils import to_categorical
import numpy as np

def PrimaryCap(inputs, dim_capsule, n_channels, kernel_size, strides, padding):
    output = layers.Conv2D(filters=dim_capsule * n_channels, kernel_size=kernel_size, strides=strides, padding=padding,
                           name='primarycap_conv2d')(inputs)
    outputs = layers.Reshape(target_shape=[-1, dim_capsule], name='primarycap_reshape')(output)
    return layers.Lambda(squash, name='primarycap_squash')(outputs)

def CapsuleLayer(num_capsule, dim_capsule, routings, name):
    def layer(input):
        input_expand = K.expand_dims(input, 2)
        input_tiled = K.tile(input_expand, [1, 1, num_capsule, 1])
        input_tiled = K.reshape(input_tiled, [-1, input.shape[1], num_capsule, input.shape[2]])
        b = K.zeros(shape=[K.shape(input)[0], input.shape[1], num_capsule, 1])
        for i in range(routings):
            c = K.softmax(b, axis=2)
            outputs = K.batch_dot(c, input_tiled, [2, 1])
            if i < routings - 1:
                b += K.batch_dot(outputs, input_tiled, [2, 3])
        return K.squeeze(outputs, axis=1)
    return layer

def Length(name):
    return layers.Lambda(lambda x: K.sqrt(K.sum(K.square(x), axis=-1)), name=name)

def Mask():
    return layers.Lambda(lambda x: x[0] * K.expand_dims(x[1], -1))

def squash(vectors, axis=-1):
    s_squared_norm = K.sum(K.square(vectors), axis, keepdims=True)
    scale = s_squared_norm / (1 + s_squared_norm) / K.sqrt(s_squared_norm + K.epsilon())
    return scale * vectors

def margin_loss(y_true, y_pred):
    L = y_true * K.square(K.maximum(0., 0.9 - y_pred)) + \
        0.5 * (1 - y_true) * K.square(K.maximum(0., y_pred - 0.1))
    return K.mean(K.sum(L, 1))

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
