import pandas as pd
import tensorflow as tf
from tensorflow import keras
#from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from classes import DataGenerator
#from tensorflow.keras.layers import TextVectorization

"""def convertImage(filePath): 
    image = tf.keras.utils.load_img(filePath)
    input_arr = tf.keras.utils.img_to_array(image)
    return input_arr

#Convert train data 
trainDataDf = pd.read_csv("book30-listing-train.csv", header=None, usecols=[1,6], encoding_errors='replace')
trainDataDf[1] = 'input/224x224/' + trainDataDf[1].astype(str) 
trainDataDf[1] = trainDataDf[1].apply(convertImage)
print(trainDataDf.head())

training_labels = trainDataDf[6]
vectorizer = TextVectorization(output_mode = "binary", ngrams=2)
vectorizer.adapt(training_labels)
oneHotLabelsTrain = vectorizer(training_labels)
print(oneHotLabelsTrain)

#Convert test data 
testDataDf = pd.read_csv("book30-listing-test.csv", header=None, usecols=[1,6], encoding_errors='replace')
testDataDf[1] = 'input/224x224/' + testDataDf[1].astype(str) 
testDataDf[1] = testDataDf[1].apply(convertImage)
print(testDataDf.head())

training_labels = testDataDf[6]
vectorizer = TextVectorization(output_mode = "binary", ngrams=2)
vectorizer.adapt(training_labels)
oneHotLabelsTest = vectorizer(training_labels)
print(oneHotLabelsTest)"""

#Train file 
trainFile = open("bookcover30-labels-train.txt","r")
trainList = trainFile.readlines()
for i in range(len(trainList)-1):
    trainList[i] = trainList[i].split()
trainFile.close()

X_train_filenames = []
for i in range(len(trainList)-1): 
    X_train_filenames.append(trainList[0])

y_train = []
for i in range(len(trainList)-1): 
    y_train.append(trainList[1])

#Test file
testFile = open("bookcover30-labels-test.txt","r")
testList = testFile.readlines()
for i in range(len(testList)-1):
    testList[i] = testList[i].split()
testFile.close()

X_test_filenames = []
for i in range(len(testList)-1): 
    X_test_filenames.append(testList[0])

y_test = []
for i in range(len(testList)-1): 
    y_test.append(testList[1])

training_batch = DataGenerator(X_train_filenames, y_train, 32)
test_batch = DataGenerator(X_test_filenames, y_test, 32)

from skimage.io import imread
from skimage.transform import resize

import keras
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.utils import to_categorical

model = Sequential()

model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu',input_shape=(80,80,3)))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters = 64, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters = 128, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters = 256, kernel_size = (5,5), activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256, activation = "relu")) #Fully connected layer
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(60, activation = "relu")) #Fully connected layer
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(12, activation = "softmax")) #Classification layer or output layer

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(x=training_batch,
                   steps_per_epoch = int(3800 // 32),
                   epochs = 10,
                   verbose = 1,
                   validation_data = test_batch,
                   validation_steps = int(950 // 32))


"""


#Preprocessing


from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Normalization
"""
"""scaler = Rescaling(scale = 1.0/255)
trainData = trainDataDf[1]
trainData = trainData.map(lambda x: (scaler(x)))"""

"""normalizer = Normalization(axis=-1)
normalizer.adapt(training_data)
normalized_data = normalizer(training_data)
print(np.var(normalized_data))
print(np.mean(normalized_data))"""
 

"""train_images = tf.convert_to_tensor(trainDataDf[1])
train_labels = tf.convert_to_tensor(trainDataDf[6])"""
"""
testDataDf = pd.read_csv("book30-listing-test.csv", header=None, usecols=[1,6], encoding_errors='replace')
testDataDf[1] = '224x224/' + testDataDf[1].astype(str)"""
"""test_images = tf.convert_to_tensor(testDataDf[1])
test_labels = tf.convert_to_tensor(testDataDf[6])"""

