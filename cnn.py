import keras
from keras import backend as K
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
import tensorflow as tf
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=2)) as sess:
    K.set_session(sess)

INPUT_FOLDER = 'tsa_datasets/stage1/aps'
PREPROCESSED_DATA_FOLDER = '/tsa_datasets/preprocessed/'
STAGE1_LABELS = 'tsa_datasets/stage1/stage1_labels.csv'
batch_size      = 16
no_epoch       = 1
examplesPer     = 10#len(SUBJECT_LIST)
ts     			= 16
size_1            = 660
size_2            = 512
num_classes = 2


model_name = 'single_image_cnn.h5'



print('Getting Data...\n')

x_train = np.load('tsa_datasets/tsa-tensors/X_prep_sliced.npy')
y_train = np.load('tsa_datasets/tsa-tensors/y_prep.npy')

print('Training (X) - Shape: ')
print(x_train.shape)
print('Training (y) - Shape: ')
print(y_train.shape)


model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


