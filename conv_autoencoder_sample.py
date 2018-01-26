# Python 2.7.10
# tensorflow (1.4.1)
# Keras (2.1.3)

from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model, Sequential

model = Sequential()

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
                #nb_epoch=50,
                nb_epoch=5,
                batch_size=128,
                shuffle=True,
                validation_data=(x_test, x_test))
                #callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
                                  
import matplotlib.pyplot as plt
#matplotlib.use('TKAgg')

n = 10

# decoded_imgs = model.predict(x_test[:n])
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display reconstruction
#     ax = plt.subplot(1, n, i+1)
#     plt.imshow(decoded_imgs[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()

from keras import backend as K

encode_model = K.function([model.layers[0].input],
                           [model.layers[5].output])
encoded_imgs_tmp = encode_model([x_test[:n]])
encoded_imgs = encoded_imgs_tmp[0].reshape(n, 4, 4*8)
#print(encoded_imgs.shape)
plt.figure(figsize=(20, 4))
for i in range(n):
    # display reconstruction
    ax = plt.subplot(1, n, i+1)
    plt.imshow(encoded_imgs[i].T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

decode_model = K.function([model.layers[6].input],
                           [model.layers[12].output])
decoded_imgs = decode_model(encoded_imgs_tmp)
decoded_imgs = decoded_imgs[0].reshape(n, 28, 28)
#print(encoded_imgs.shape)
plt.figure(figsize=(20, 4))
for i in range(n):
    # display reconstruction
    ax = plt.subplot(1, n, i+1)
    plt.imshow(decoded_imgs[i].T)
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

