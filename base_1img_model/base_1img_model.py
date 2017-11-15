
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd
import tsahelper.tsahelper as tsa
import matplotlib.pyplot
import matplotlib.animation






INPUT_FOLDER = '/gpfs/scratch/spd13/tsa_datasets/stage1/aps'

STAGE1_LABELS = '/gpfs/scratch/spd13/tsa_datasets/stage1/stage1_labels.csv'





# OPTION 1: get a list of all subjects for which there are labels
df = pd.read_csv(STAGE1_LABELS)
df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
SUBJECT_LIST = df['Subject'].unique()
y_df =df.pivot_table(df,columns=[df.Zone],index=df.Subject)
y_index = y_df.index.get_values()
y_array=y_df.values





batch_size      = 16

examplesPer     = len(SUBJECT_LIST)
ts      = 16

size_1            = 660
size_2            = 512





print('Loading y')
y =np.load('/storage/work/spd13/TSA/y_prep.npy')
print('Loading X.....')
X = np.load('/storage/work/spd13/TSA/X_prep.npy')
y_unroll = np.zeros((18352,17))
for i in range(y.shape[0]):
    for j in range(0,16):
        y_unroll[i*j,:] = y[i,:]
        
X_unroll=X.reshape(18352,660,512,1)
#if you want to train only for 1 image from the sequence use 
#X_1 = X[:,1,:,:,:]







from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D ,TimeDistributed
from keras.layers import Activation, Dropout, Flatten, Dense 
import keras.layers
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.layers import BatchNormalization
from keras.optimizers import RMSprop, Adadelta


import pydot
import tensorflow as tf
from keras import backend as K
server =tf.train.Server.create_local_server()
sess = tf.Session(server.target)
with tf.Session(config=tf.ConfigProto(
                    intra_op_parallelism_threads=160)) as sess:
    K.set_session(sess)

data_augmentation=False





model = Sequential()
model.add(Conv2D(96, 11,strides=4,padding='valid', input_shape=(660, 512,1), data_format='channels_last', activation='relu'))

model.add((MaxPooling2D(pool_size=(3,3),strides=2,padding ='valid')))
model.add((Conv2D(96, 5,padding='valid',activation='relu')))

model.add((MaxPooling2D(pool_size=(3,3),strides=2,padding ='valid')))
model.add((BatchNormalization()))
model.add((Conv2D(96, 3,padding='valid',activation='relu')))
model.add((Conv2D(96, 3,padding='valid',activation='relu')))
model.add((MaxPooling2D(pool_size=(3,3),strides=2,padding ='valid')))
model.add((BatchNormalization()))

model.add((Conv2D(48, 3,padding='valid',activation='relu')))
model.add((Conv2D(48, 3,padding='valid',activation='relu')))
model.add((MaxPooling2D(pool_size=(3,3),strides=2,padding ='valid')))
model.add((BatchNormalization()))


model.add((Flatten()))
model.add((Dense(360,activation='relu')))
model.add(Dropout(0.2))
model.add((Dense(180,activation='relu')))

model.add(Dropout(0.2))
model.add(Dense(17,activation='sigmoid'))
model.add(Dense(17,activation='sigmoid'))
model.summary()

rmsprop = RMSprop(lr=0.01, epsilon=1e-08, decay=0.0)
model.compile(loss='binary_crossentropy', optimizer=rmsprop , metrics=['accuracy'])

earlystop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=3, verbose=1, mode='auto')


from keras.utils import plot_model
#plot_model(model, to_file='model.png')
tf_board=keras.callbacks.TensorBoard(log_dir='./logs',
                                     histogram_freq=16,
                                     batch_size=16,
                                     write_graph=True,
                                     write_grads=False,
                                     write_images=True,
                                     embeddings_freq=0,
                                     embeddings_layer_names=None,
                                     embeddings_metadata=None)
filepath="weights.best.hdf5"
from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True, mode='max')


### If training for one image use X_1,y
model.fit(X_unroll,
          y_unroll,batch_size=batch_size,
          nb_epoch=20,verbose=1,
          callbacks=[tf_board,checkpoint,earlystop],
          validation_split=0.2,
          #validation_data=(X_cv,y_cv),
          #shuffle=True
             )


model.save('simple_cnn_1_model.h5')




# # Submisson Routine




# # Reading Submission file

print('Preping Submission file')

submission = pd.read_csv('stage1_sample_submission.csv')

submission['Subject'], submission['Zone'] = submission['Id'].str.split('_',1).str
TEST_SUBJECT_LIST = submission['Subject'].unique()


# # Submisson Routine

print('Model Predictions to csv file ')


test_examples= len(TEST_SUBJECT_LIST)
  
X_1_test    = np.load('/storage/work/spd13/TSA/X_1_test_prep.npy')

preds   = model.predict(X_1_test)
submission['Probability'] = preds.flatten()
submission = submission.drop(["Subject","Zone"],axis=1)
submission.index = submission.Id
submission = submission.drop(["Id"],axis=1)
submission.to_csv('submission.csv')




