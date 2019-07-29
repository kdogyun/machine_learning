import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Batch Normalization

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.BatchNormalization()) # 배치 정규화 추가
model.add(keras.layers.Dense(128, kernel_initializer='he_normal')) # He 초기화 추가
# 해 초기화는 어디어디에 붙여야하는거지?
# model.add(keras.layers.Dense(128))
model.add(keras.layers.BatchNormalization()) # 배치 정규화 추가
model.add(keras.layers.Activation("relu"))
model.add(keras.layers.Dense(10))
model.add(keras.layers.BatchNormalization()) # 배치 정규화 추가
model.add(keras.layers.Activation("softmax"))
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])
# 메트릭스에 어큐레이시로 안주고 다른걸로 주면 정확도 0.1;;
# he 초기화 안줬을때 0.8834 / 해줬을때 0.8753
# 왜지??

history = model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
