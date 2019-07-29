import tensorflow as tf
from tensorflow import keras
import numpy as np

# Data Augmentation (3가지 기법 이상 적용)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = train_images.reshape(len(train_images), 28, 28, 1)
test_images = test_images.reshape(len(test_images), 28, 28, 1)

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


datagen = keras.preprocessing.image.ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)
# https://keras.io/preprocessing/image/

datagen.fit(train_images)
history = model.fit_generator(datagen.flow(train_images, train_labels, batch_size=32),
                              epochs=10, steps_per_epoch=train_images.shape[0] / 32)
#  공식홈페이지 예제 참조
# 에폭 5: 학습은 loss 0.7407 acc 0.7215 / 테스트 샘플은 loss 1.4071 acc 0.4301
# 에폭 10: 학습은 loss 0.7054 acc 0.7344 / 테스트 샘플은 loss 1.3949 acc 0.4734
# steps_per_epoch 이게 에폭별 실행 횟수 인건가?

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
