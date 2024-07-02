# dataset.py
from keras.datasets import mnist
from keras.utils import to_categorical

def load_data():
    # Load the MNIST dataset from Keras
    (train_data, train_labels), (test_data, test_labels) = mnist.load_data()

    # Preprocess the data
    train_data = train_data.astype('float32') / 255.0
    test_data = test_data.astype('float32') / 255.0
    train_data = train_data[..., None]
    test_data = test_data[..., None]
    train_labels = to_categorical(train_labels, 10)
    test_labels = to_categorical(test_labels, 10)

    return train_data, train_labels, test_data, test_labels
