import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# Ensemble (Bagging - 3 Models)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def my_model():
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28), kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu, kernel_initializer='he_normal'))
    model.add(keras.layers.Dense(10, activation=tf.nn.softmax, kernel_initializer='he_normal'))
    model.compile(loss='categorical_crossentropy',
                 optimizer='adam',
                 metrics=[keras.metrics.categorical_accuracy])
    return model

model1 = KerasClassifier(build_fn = my_model, epochs = 5, verbose = 0)
model2 = KerasClassifier(build_fn = my_model, epochs = 5, verbose = 0)
model3 = KerasClassifier(build_fn = my_model, epochs = 5, verbose = 0)

ensemble_clf = VotingClassifier(estimators = [('model1', model1), ('model2', model2), ('model3', model3)], voting = 'soft')
ensemble_clf.fit(train_images, train_labels)
y_pred = ensemble_clf.predict(test_images)

print('Test accuracy:', accuracy_score(y_pred, test_labels))
