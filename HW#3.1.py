import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import random

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0 # 학습시키려면 0~1사이 값이어야하므로
test_images = test_images / 255.0 # 학습시키려면 0~1사이 값이어야하므로

class_name = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28))) # 입력레이어
# 2차원 행렬을 1차원으로 펴주는 함수
model.add(keras.layers.Dense(128, activation=tf.nn.relu)) # 은닉레이어
model.add(keras.layers.Dense(10, activation=tf.nn.softmax)) # 출력레이어
# 출력이 큰건 크게 작은건 작게:softmax

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adam(lr=1e0),
              metrics=['accuracy'])
# 컴파일: 손실함수(Loss function), 최적화(Optimizer), 평가지표(Metrics)
# 메트릭스를 안주면 안되나? 정확도 말고 다른건 쓸 수 없나?

history = model.fit(train_images, train_labels, epochs=5, batch_size=64)
# 모델 학습
# history에 담아서 어디다가 쓰는거지?

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
# 학습 완료후, 모델의 정확도 찍어보기

predictions = model.predict(test_images) # 테스트 이미지로 예측 구하기
# np.argmax(predictions[0]) # max 인덱스 구하기

for i in range(15):
    a = random.randrange(len(test_images)) # 테스트 이미지 중에 랜덤하게 고르기
    
    plt.subplot(5, 6, 2*i +1)
    plt.imshow(test_image[a]) # plot에 이미지 그리기

    label_pred = np.argmax(predictions[a])
    label_true = test_labels[a]

    if label_pred == label_true:
        plt.xlabel("{}".format(class_names[label_pred]), color='blue') # 포맷을 이용해 더 이쁘게
    else:
        plt.xlabel("{} ({})".format(class_names[label_pred], class_names[label_true]), color='red')
        

    plt.subplot(5, 6, 2 * i + 2)
    plt.bar(range(10), predictions[a], color='gray')
    plt.bar(label_true, predictions[a][label_true], color='b')
    if label_pred != true_label:
        plt.bar(label_pred, predictions[a][label_pred], color='r')

plt.show()






    
    
