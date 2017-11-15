import numpy as np 
import pandas as pd
import tsahelper.tsahelper as tsa
import matplotlib.pyplot
import matplotlib.animation


INPUT_FOLDER = '/gpfs/scratch/spd13/tsa_datasets/stage1/aps'
PREPROCESSED_DATA_FOLDER = '/gpfs/scratch/spd13/tsa_datasets/preprocessed/'
STAGE1_LABELS = '/gpfs/scratch/spd13/tsa_datasets/stage1/stage1_labels.csv'

# OPTION 1: get a list of all subjects for which there are labels
df = pd.read_csv(STAGE1_LABELS)
df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
SUBJECT_LIST = df['Subject'].unique()

y_df =df.pivot_table(df,columns=[df.Zone],index=df.Subject)

batch_size      = 32
nb_epochs       = 1
examplesPer     = 100
ts      = 16
hidden_units    = 200
size_1            = 660
size_2            = 512

#run epochs of sampling data then training
for ep in range(0,nb_epochs):
    X_train       = []
    y_train       = []  
    
    X_train     = np.zeros((examplesPer,ts,size_1,size_2,1))

    for i in range(0,examplesPer):
        #initialize a training example of max_num_time_steps,im_size,im_size
        output      = np.zeros((ts,size_1,size_2,1))
        #sum up the outputs for new output
        ff = y_df.ix[SUBJECT_LIST[i]]
        ff = ff.values.reshape(17)
        exampleY    = ff
        output[0:16,:,:,0] = tsa.read_data(INPUT_FOLDER+'/'+SUBJECT_LIST[i]+'.aps').transpose()
        X_train[i,:,:,:,:] = output
        y_train.append(exampleY)

    y_train     = np.array(y_train)
    
    if ep == 0:
        print("X_train shape: ",X_train.shape)
        print("y_train shape: ",y_train.shape)
        
#Building the model 

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense 
import keras.layers
from keras.layers.recurrent import SimpleRNN, LSTM, GRU
from keras.optimizers import RMSprop, Adadelta


model = Sequential()
model.add(keras.layers.TimeDistributed(Conv2D(16, 4,strides=4,padding='valid'), input_shape=(16, 660, 512,1)))
model.add(Activation('relu'))
model.add(keras.layers.TimeDistributed(Conv2D(32, 3,strides=3,padding='valid')))
model.add(keras.layers.TimeDistributed(MaxPooling2D(pool_size=(3,3),padding ='valid')))
model.add(Activation('relu'))
model.add(keras.layers.TimeDistributed(Flatten()))
model.add(Activation('relu'))
model.add(GRU(output_dim=100,return_sequences=True))
model.add(GRU(output_dim=50,return_sequences=False))
model.add(Dropout(.2))
model.add(Dense(17))

rmsprop = RMSprop()
model.compile(loss='mean_squared_error', optimizer=rmsprop)

#Train

model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=1,verbose=1)

#TEST

X_test        = []
y_test        = [] 
test_examples= 50
X_test     = np.zeros((test_examples,ts,size_1,size_2,1))
indices     = np.random.choice(range(0,1000),size=50)
for i in range(0,test_examples):
        #initialize a training example of max_num_time_steps,im_size,im_size
        output      = np.zeros((ts,size_1,size_2,1))
        #sum up the outputs for new output
        index = indices[i]
        ff = y_df.ix[SUBJECT_LIST[i]]
        ff = ff.values.reshape(17)
        exampleY    = ff
        output[0:16,:,:,0] = tsa.read_data(INPUT_FOLDER+'/'+SUBJECT_LIST[i]+'.aps').transpose()
        X_test[i,:,:,:,:] = output
        y_test.append(exampleY)


X_test  = np.array(X_test)
y_test  = np.array(y_test)       
preds   = model.predict(X_test)

print("X_test shape: ",X_test.shape)
print("y_test shape: ",y_test.shape)

#Eval
#from sklearn.metrics import accuracy_score
#accuracy_score(y_test[:,0],preds[:,0])
