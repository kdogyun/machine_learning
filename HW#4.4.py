import tensorflow as tf
from tensorflow import keras
import numpy as np

# Data Augmentation (3가지 기법 이상 적용)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
train_images = train_images.values.reshape(-1,28,28,1)
test_images = test_images.values.reshape(-1,28,28,1)
# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
train_labels = to_categorical(train_labels, num_classes = 10)
test_labels = to_categorical(test_labels, num_classes = 10)

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28), kernel_initializer='he_normal'))
model.add(keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_normal'))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='he_normal'))
model.compile(loss='categorical_crossentropy',
             optimizer='adam',
             metrics=[keras.metrics.categorical_accuracy])


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

datagen.fit(train_images)

history = model.fit_generator(datagen.flow(train_images,train_labels, batch_size=64),
                              epochs = 5, verbose = 2, steps_per_epoch=train_images.shape[0] // 64)
#  validation_data = (test_images, test_labels),

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
