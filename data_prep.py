import numpy as np 
import pandas as pd
import tsahelper.tsahelper as tsa
import matplotlib.pyplot
import matplotlib.animation
from numba import jit, prange



#INPUT_FOLDER = '/gpfs/scratch/spd13/tsa_datasets/stage1/aps'
#PREPROCESSED_DATA_FOLDER = '/gpfs/scratch/spd13/tsa_datasets/preprocessed/'
#STAGE1_LABELS = '/gpfs/scratch/spd13/tsa_datasets/stage1/stage1_labels.csv'

INPUT_FOLDER = 'tsa_datasets/stage1/aps'
PREPROCESSED_DATA_FOLDER = '/tsa_datasets/preprocessed/'
STAGE1_LABELS = 'tsa_datasets/stage1/stage1_labels.csv'



# OPTION 1: get a list of all subjects for which there are labels
df = pd.read_csv(STAGE1_LABELS)
df['Subject'], df['Zone'] = df['Id'].str.split('_',1).str
SUBJECT_LIST = df['Subject'].unique()
y_df =df.pivot_table(df,columns=[df.Zone],index=df.Subject)
y_index = y_df.index.get_values()
y_array=y_df.values

batch_size      = 16
no_epoch       = 100
examplesPer     = 10 #len(SUBJECT_LIST)
ts     			= 16
size_1            = 660
size_2            = 512
print('DATA Prep')
#run epochs of sampling data then training

#X_train       = []
#y_train       = []  
#X_train     = np.zeros((examplesPer,ts,size_1,size_2,1))
@jit(      #__________________ a list of signatures for prepared alternative code-paths, to avoid a deferred lazy-compilation if undefined
        nopython = False,      #__________________ forces the function to be compiled in nopython mode. If not possible, compilation will raise an error.
        nogil    = True,      #__________________ tries to release the global interpreter lock inside the compiled function. The GIL will only be released if Numba can compile the function in nopython mode, otherwise a compilation warning will be printed.
        cache    = False,      #__________________ enables a file-based cache to shorten compilation times when the function was already compiled in a previous invocation. The cache is maintained in the __pycache__ subdirectory of the directory containing the source file.
        forceobj = False,      #__________________ forces the function to be compiled in object mode. Since object mode is slower than nopython mode, this is mostly useful for testing purposes.
        locals   = {}          #__________________ a mapping of local variable names to Numba Types.
        ) #____________________# [_v41] ZERO <____ TEST *ALL* CALLED sub-func()-s to @.jit() too >>>>>>>>>>>>>>>>>>>>> [DONE]

#for i in range(0,examplesPer):
def transfrom_data(SUBJECT_LIST,get_slice = False):
    print('Parallel Data Processing')

    y_train       = np.zeros((examplesPer,17))
    X_train     = np.zeros((examplesPer,ts,size_1,size_2,1))
    for i in prange(examplesPer):
        output      = np.zeros((ts,size_1,size_2,1))
        index=np.where(y_index==SUBJECT_LIST[i])
        ff = y_array[index]
        #ff = ff.values.reshape(17)
        exampleY    = ff
        output[0:ts,:,:,0] = tsa.read_data(INPUT_FOLDER+'/'+SUBJECT_LIST[i]+'.aps').transpose()
        X_train[i,:,:,:,:] = output
        y_train[i,:] = exampleY

    if get_slice:
        X_train = X_train[:,0,:,:,:]

    print('DATA Prep : Done')
    return X_train,y_train

 
 
X_train,y_train = transfrom_data(SUBJECT_LIST)



print("X_train shape: ",X_train.shape)
print("y_train shape: ",y_train.shape)
print('Saving X')
np.save('tsa_datasets/tsa-tensors/X_prep.npy', X_train)
print('Saving y')
np.save('tsa_datasets/tsa-tensors/y_prep.npy', y_train)

## Get Test Data
print('Getting Test Data from Submission file')

submission = pd.read_csv('tsa_datasets/stage1/stage1_sample_submission.csv')

submission['Subject'], submission['Zone'] = submission['Id'].str.split('_',1).str
TEST_SUBJECT_LIST = submission['Subject'].unique()


## Transform test data into a tensor

@jit(      #__________________ a list of signatures for prepared alternative code-paths, to avoid a deferred lazy-compilation if undefined
        nopython = False,      #__________________ forces the function to be compiled in nopython mode. If not possible, compilation will raise an error.
        nogil    = True,      #__________________ tries to release the global interpreter lock inside the compiled function. The GIL will only be released if Numba can compile the function in nopython mode, otherwise a compilation warning will be printed.
        cache    = False,      #__________________ enables a file-based cache to shorten compilation times when the function was already compiled in a previous invocation. The cache is maintained in the __pycache__ subdirectory of the directory containing the source file.
        forceobj = False,      #__________________ forces the function to be compiled in object mode. Since object mode is slower than nopython mode, this is mostly useful for testing purposes.
        locals   = {}          #__________________ a mapping of local variable names to Numba Types.
        ) #____________________# [_v41] ZERO <____ TEST *ALL* CALLED sub-func()-s to @.jit() too >>>>>>>>>>>>>>>>>>>>> [DONE]

def transform_test_data(test_subjects_list,get_slice = False):
    print("Transforming Test Data")
    test_examples= len(test_subjects_list)
    X_test = np.zeros((test_examples,ts,size_1,size_2,1))
    for i in prange(0,test_examples):
            #initialize a training example of max_num_time_steps,im_size,im_size
            output = np.zeros((ts,size_1,size_2,1))
            #sum up the outputs for new output
            output[0:16,:,:,0] = tsa.read_data(INPUT_FOLDER+'/'+TEST_SUBJECT_LIST[i]+'.aps').transpose()
            X_test[i,:,:,:,:] = output

    if get_slice:
        X_test = X_test[:,0,:,:,:]
    return X_test

X_test = transform_test_data(TEST_SUBJECT_LIST)

print("X_test shape: ",X_test.shape)
print('Saving X_test')
np.save('tsa_datasets/tsa-tensors/X_test.npy', X_test)


## Get tensor with single image from each subject

X_train_sliced,y_train = transfrom_data(SUBJECT_LIST,get_slice=True)
print("X_train_sliced shape: ",X_train_sliced.shape)
print('Saving X w/ slice ')
np.save('tsa_datasets/tsa-tensors/X_prep_sliced.npy', X_train_sliced)

X_test_sliced = transform_test_data(TEST_SUBJECT_LIST,get_slice=True)

print("X_test_sliced shape: ",X_test_sliced.shape)
print('Saving X_test_sliced')
np.save('tsa_datasets/tsa-tensors/X_test_sliced.npy', X_test_sliced)

