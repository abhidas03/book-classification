import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def convertImage(filePath):
    image = tf.keras.utils.load_img(filePath)
    input_arr = tf.keras.utils.img_to_array(image)
    return input_arr


#Convert train data
trainDataDf = pd.read_csv("input/book30-listing-train.csv", header=None, usecols=[1,6], encoding_errors='replace')
trainDataDfLess = trainDataDf.head(5000)
testDataDf = pd.read_csv("input/book30-listing-test.csv", header=None, usecols=[1,6], encoding_errors='replace')
testDataDfLess = testDataDf.head(5000)

trainDataDfLess.iloc[:, 0] = 'input/224x224/' + trainDataDfLess.iloc[:, 0]
trainDataDfLess.iloc[:, 0] = trainDataDfLess.iloc[:, 0].apply(convertImage)

testDataDfLess.iloc[:, 0] = 'input/224x224/' + testDataDfLess.iloc[:, 0]
testDataDfLess.iloc[:, 0] = testDataDfLess.iloc[:, 0].apply(convertImage)

#Preprocessing
X_train = trainDataDfLess[1]
X_test = testDataDfLess[1]


y_train = trainDataDfLess[6]
y_test = testDataDfLess[6]


X_train=X_train/255
X_test=X_test/255


from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = to_categorical(label_encoder.fit_transform(y_train), num_classes=30)
y_test_encoded = to_categorical(label_encoder.transform(y_test), num_classes=30)
print(y_train_encoded.shape)
print(y_test_encoded.shape)
print(y_train_encoded[0])


X_train = np.array([x for x in X_train])
X_train = X_train.reshape(-1, 224, 224, 3)
X_train = X_train.astype('float32')


X_test = np.array([x for x in X_test])
X_test = X_test.reshape(-1, 224, 224, 3)
X_test = X_test.astype('float32')

from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

print(X_train.shape)
print(X_test.shape)
model = Sequential([
    Conv2D(filters=32,kernel_size=(3, 3), padding='same', activation='relu',input_shape = (224, 224, 3)),
    Conv2D(filters=32,kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=2) ,
    Dropout(0.25),
    Conv2D(filters=32,kernel_size=(3, 3), padding='same', activation='relu'),
    Conv2D(filters=32,kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=2),
    Dropout(0.25),
    Flatten(), # flatten out the layers
    Dense(256,activation='relu'),
    Dense(256,activation='relu'),
    Dense(30,activation = 'softmax')
])
print(model.summary())

batch_size = 32
epochs = 3

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(X_train, y_train_encoded, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data=(X_test, y_test_encoded))


score = model.evaluate(X_test, y_test_encoded, verbose=0)
print("Test accuracy:", score[1]*100)



