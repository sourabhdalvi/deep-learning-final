
import numpy as np 
import pandas as pd
import tsahelper.tsahelper as tsa
import matplotlib.pyplot
import matplotlib.animation
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
import keras.layers
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop, Adadelta
import tensorflow as tf
from keras import backend as K
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
print('Getting Data...')

X_train = np.load('tsa_datasets/tsa-tensors/X_prep.npy')
y_train = np.load('tsa_datasets/tsa-tensors/y_prep.npy')

print('Building Model')
model = Sequential()
model.add(keras.layers.TimeDistributed(Conv2D(96, 11,strides=4,padding='valid'), input_shape=(16, 660, 512,1)))
model.add(Activation('relu'))
model.add(keras.layers.TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=2,padding ='valid')))
model.add(keras.layers.TimeDistributed(Conv2D(256, 5,padding='valid')))
model.add(Activation('relu'))
model.add(keras.layers.TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=2,padding ='valid')))
model.add(keras.layers.TimeDistributed(BatchNormalization()))
model.add(keras.layers.TimeDistributed(Conv2D(384, 3,padding='valid')))
model.add(Activation('relu'))
model.add(keras.layers.TimeDistributed(Conv2D(384, 3,padding='valid')))
model.add(Activation('relu'))
model.add(keras.layers.TimeDistributed(Conv2D(256, 3,padding='valid')))
model.add(Activation('relu'))
model.add(keras.layers.TimeDistributed(MaxPooling2D(pool_size=(3,3),strides=2,padding ='valid')))
model.add(keras.layers.TimeDistributed(BatchNormalization()))
model.add(keras.layers.TimeDistributed(Dense(4096)))
model.add(Activation('tanh'))
model.add(Dropout(0.5))

# This adds recurrence.
model.add(keras.layers.TimeDistributed(Flatten()))
model.add(Activation('relu'))
model.add(GRU(units=100,return_sequences=True))
model.add(GRU(units = 50,return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(units =34, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units =17, activation='softmax'))

rmsprop = RMSprop()
model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['categorical_accuracy'])


print('Training Model')
model.fit(X_train, y_train, batch_size=batch_size, epochs=no_epoch,verbose=1)

## Get Test Data
print("Getting Test Data: ")

# Submisson Routine

def pred_to_submission(predictions):
    print('Preping Submission file')

    submission = pd.read_csv('tsa_datasets/stage1/stage1_sample_submission.csv')

    submission['Subject'], submission['Zone'] = submission['Id'].str.split('_', 1).str

    submission['Probability'] = predictions.flatten()
    submission = submission.drop(["Subject", "Zone"], axis=1)
    submission.index = submission.Id
    submission = submission.drop(["Id"], axis=1)
    return submission



X_test = np.load('tsa_datasets/tsa-tensors/X_test.npy')
print("Making Predictions on Test Data")
preds = model.predict(X_test)
submission = pred_to_submission(predictions=preds)
print('Model Predictions to csv file ')
submission.to_csv('crnn-submission.csv')
print("-----Complete!-----")

