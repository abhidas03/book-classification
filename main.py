import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from classes import DataGenerator

def convertImage(filePath): 
    image = tf.keras.utils.load_img(filePath)
    input_arr = tf.keras.utils.img_to_array(image)
    return input_arr

#Convert train data 
trainDataDf = pd.read_csv("book30-listing-train.csv", header=None, usecols=[1,6], encoding_errors='replace')
trainDataDf[1] = 'input/224x224/' + trainDataDf[1].astype(str) 
trainDataDf[1] = trainDataDf[1].apply(convertImage)
print(trainDataDf.head())

from tensorflow.keras.layers import TextVectorization
training_labels = trainDataDf[6]
vectorizer = TextVectorization(output_mode = "binary", ngrams=2)
vectorizer.adapt(training_labels)
oneHotLabels = vectorizer(training_labels)
print(oneHotLabels)

# Parameters
params = {'dim': (32,32,32),
          'batch_size': 64,
          'n_classes': 6,
          'n_channels': 1,
          'shuffle': True}

# Datasets
partition = trainDataDf[1]
labels = oneHotLabels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

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

