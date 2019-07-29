import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Batch Normalization

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28), kernel_initializer='he_normal')) # He 가중치 초기화 추가
model.add(BatchNormalization()) # 배치 정규화 추가
model.add(keras.layers.Dense(128, kernel_initializer='he_normal')) # He 초기화 추가
model.add(BatchNormalization()) # 배치 정규화 추가
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(10, kernel_initializer='he_normal')) # He 초기화 추가
model.add(BatchNormalization()) # 배치 정규화 추가
model.add(keras.layers.Activation("softmax"))
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=[keras.metrics.categorical_accuracy])
history = model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
