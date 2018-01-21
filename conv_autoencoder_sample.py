from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, Sequential

model = Sequential()
#input_img = Input(shape=(1, 28, 28))

model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(28,28,1)))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), padding='same'))

model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(UpSampling2D((2, 2)))
model.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

#autoencoder = Model(input_img, decoded)
model.compile(optimizer='adam', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))
x_test = np.reshape(x_test, (len(x_test), 28, 28, 1))

model.fit(x_train, x_train,
                nb_epoch=50,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
                #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

import matplotlib.pyplot as plt

n = 10
#encoder = Model(input_img, encoded)
decoded_imgs = model.predict(x_test[:n])

plt.figure(figsize=(20, 8))
for i in range(n):
    for j in range(8):
        ax = plt.subplot(8, n, j*n + i+1)
        plt.imshow(decoded_imgs[i][j], interpolation='none')
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
plt.show()
