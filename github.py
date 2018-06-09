#importing the libraries
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import keras
import h5py
import scipy.io
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils.np_utils import to_categorical
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score,mean_squared_error,mean_absolute_error)
from keras.layers import Convolution1D, Flatten, MaxPooling1D
import matplotlib.pyplot as plt
from keras.layers.advanced_activations import PReLU
from keras import callbacks
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

#load the data
Ntotal = 100
Ntest=20
Ntrain =Ntotal-Ntest
time =4096
dataset_O = np.empty((Ntotal, time))
dataset_S = np.empty((Ntotal, time))
noise=np.empty((Ntotal, time))

for path, subdirs, files in os.walk('./N/'):
    for i in range(Ntotal):
        dataset_train_tempO = pd.read_csv('./N/' + files[i])
        dataset_train_tempO = np.array(dataset_train_tempO, dtype = 'float')
        dataset_O[i] = dataset_train_tempO[0:time, 0]
        
        
for path, subdirs, files in os.walk('./S/'):
    for i in range(Ntotal):
        dataset_train_tempS = pd.read_csv('./S/' + files[i])
        dataset_train_tempS = np.array(dataset_train_tempS, dtype = 'float')
        dataset_S[i] = dataset_train_tempS[0:time, 0]

# Standardization
def Norm_Data(inputData):
    dataScaled = np.empty(inputData.shape)
    for i in range(inputData.shape[0]):
        tempMean = np.mean(inputData[i,:])
        tempStd = np.std(inputData[i,:])
        tempMin= np.min(inputData[i,:])
        tempMax= np.max(inputData[i,:])
        dataScaled[i,:] = (inputData[i,:] - tempMean) / tempStd
#        dataScaled[i,:] = (inputData[i,:] - tempMin) / (tempMax-tempMin)
    return dataScaled

def get_confusion_matrix_one_hot(model_results, truth):
   
    assert model_results.shape == truth.shape
    num_outputs = truth.shape[1]
    confusion_matrix = np.zeros((num_outputs, num_outputs), dtype=np.int32)
    predictions = np.argmax(model_results,axis=1)
    assert len(predictions)==truth.shape[0]

    for actual_class in range(num_outputs):
        idx_examples_this_class = truth[:,actual_class]==1
        prediction_for_this_class = predictions[idx_examples_this_class]
        for predicted_class in range(num_outputs):
            count = np.sum(prediction_for_this_class==predicted_class)
            confusion_matrix[actual_class, predicted_class] = count
    assert np.sum(confusion_matrix)==len(truth)
    assert np.sum(confusion_matrix)==np.sum(truth)
    return confusion_matrix
        
dataset_o = Norm_Data(dataset_O)
dataset_s = Norm_Data(dataset_S)
                          
dataset_train_o = dataset_o[0:Ntrain]
dataset_train_s = dataset_s[0:Ntrain]

label_train_o = np.empty((Ntrain))
label_train_s = np.empty((Ntrain))

label_train_o[:] = 0
label_train_s[:] = 1

label_train_o = keras.utils.to_categorical(label_train_o, 2)
label_train_s = keras.utils.to_categorical(label_train_s, 2)

combined_dataset_train = np.concatenate((dataset_train_o, dataset_train_s), axis=0)
combined_label_train= np.concatenate((label_train_o, label_train_s), axis=0)

dataset_test_o = dataset_o[Ntotal-Ntest:Ntotal]
dataset_test_s = dataset_s[Ntotal-Ntest:Ntotal]

label_test_o = np.empty((Ntest))
label_test_s = np.empty((Ntest))

label_test_o[:] = 0
label_test_s[:] = 1

label_test_o = keras.utils.to_categorical(label_test_o, 2)
label_test_s = keras.utils.to_categorical(label_test_s, 2)

combined_dataset_test = np.concatenate((dataset_test_o, dataset_test_s), axis=0)
combined_label_test= np.concatenate((label_test_o, label_test_s,), axis=0)

combined_dataset_train = combined_dataset_train.reshape(combined_dataset_train.shape[0], time, 1)
combined_dataset_test = combined_dataset_test.reshape(combined_dataset_test.shape[0], time, 1)

inputShape = (time, 1)
batch_size =3


#.define network 
model = Sequential()
model.add(Convolution1D(5,9,strides = 1, activation='relu', input_shape=inputShape))
model.add(MaxPooling1D(pool_length=(2), strides = 2))
model.add(Dropout(0.2))
model.add(Convolution1D(10,7,strides = 2, activation='relu'))
model.add(MaxPooling1D(pool_length=(2), strides = 2))
model.add(Dropout(0.2))
model.add(Convolution1D(15,5,strides = 1, activation='relu'))
model.add(MaxPooling1D(pool_length=(2), strides = 2))
model.add(Dropout(0.2))
model.add(Convolution1D(20,3,strides = 2, activation='relu'))
model.add(MaxPooling1D(pool_length=(2), strides = 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))
model.summary()

opt=optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
                      
model.compile(loss='categorical_crossentropy',optimizer=opt, metrics=['accuracy'])


# Saving best weights only. Overlapping.
checkpointer = callbacks.ModelCheckpoint(filepath="logs/cnn1layer/weights_best.hdf5",
                                         verbose=1, save_best_only=True, monitor='val_loss', mode='auto')
history=model.fit(combined_dataset_train, combined_label_train, batch_size=batch_size,
          nb_epoch=50, callbacks=[checkpointer], shuffle=True, validation_split = 0.3)

model.save("logs/cnn1layer/cnn1layer_model.hdf5")
# Saving last weights
model.save_weights("logs/cnn1layer/weights_last.hdf5")

loss, accuracy = model.evaluate(combined_dataset_test, combined_label_test)
print("\nOn last: \n Loss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
print("--------------------------------------------------------------------")
predictions_last = model.predict(combined_dataset_test)
scipy.io.savemat('total_value.mat', dict(total_pred_val=predictions_last, total_orig_label = combined_label_test))
cnf_matrix_last=get_confusion_matrix_one_hot(predictions_last, combined_label_test)
total1=sum(sum(cnf_matrix_last))
accuracy1=(cnf_matrix_last[0,0]+cnf_matrix_last[1,1])/total1
print ('Accuracy_Last : ', accuracy1)

sensitivity = cnf_matrix_last[1,1]/(cnf_matrix_last[1,1]+cnf_matrix_last[1,0])
print('Sensitivity_last : ', sensitivity )

specificity = cnf_matrix_last[0,0]/(cnf_matrix_last[0,1]+cnf_matrix_last[0,0])
print('Specificity_last : ', specificity)


model.load_weights("logs/cnn1layer/weights_best.hdf5")
loss1, accuracy1 = model.evaluate(combined_dataset_test, combined_label_test)
print("\n On Best: \n Loss: %.2f, Accuracy: %.2f%%" % (loss1, accuracy1*100))
predictions_best = model.predict(combined_dataset_test)
cnf_matrix=get_confusion_matrix_one_hot(predictions_best, combined_label_test)
total1=sum(sum(cnf_matrix))
accuracy1=(cnf_matrix[0,0]+cnf_matrix[1,1])/total1
print ('Accuracy_best : ', accuracy1)

sensitivity1 = cnf_matrix[1,1]/(cnf_matrix[1,1]+cnf_matrix[1,0])
print('Sensitivity_best : ', sensitivity1 )

specificity1 = cnf_matrix[0,0]/(cnf_matrix[0,0]+cnf_matrix[0,1])
print('Specificity_best : ', specificity1)

