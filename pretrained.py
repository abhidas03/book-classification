import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout
import keras.models
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd
#VGG16 - Pretrained model
vgg = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(224, 224, 3), include_top=False, classes=30)
for layer in vgg.layers:
    layer.trainable = False

trainDataDf = pd.read_csv("input/bookcover30-labels-train.csv", header=None, encoding_errors='replace')
testDataDf = pd.read_csv("input/bookcover30-labels-test.csv", header=None, encoding_errors='replace')

#Get correct file path
trainDataDf.iloc[:, 0] = 'input/224x224/' + trainDataDf.iloc[:, 0]
testDataDf.iloc[:, 0] = 'input/224x224/' + testDataDf.iloc[:, 0]

#Label columns for later
trainDataDf.columns = ['file', 'genre']
testDataDf.columns = ['file', 'genre']

print(trainDataDf.head())
print(testDataDf.head())

#Flatten output of vgg model

x = vgg.output
x = BatchNormalization()(x)
x = Dense(256, activation ='relu')(x)
x = Dropout(0.7)(x)
x = Flatten()(x)

#Combine vgg model with a output layer
prediction = Dense(30, activation='softmax')(x)

#Make model
model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

from keras import optimizers

#Using legacy adam bc of recent errors with Mac M1/M2 (use regular fixed)

optimizer = optimizers.Adam(learning_rate=0.0005)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=False)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_set = train_datagen.flow_from_dataframe(trainDataDf,
                                              target_size=(224, 224),
                                              batch_size= 64,
                                              class_mode='categorical',
                                              x_col='file',
                                              y_col='genre')

test_set = test_datagen.flow_from_dataframe(testDataDf,
                                            target_size=(224, 224),
                                            batch_size= 64,
                                            class_mode='categorical',
                                            x_col='file',
                                            y_col='genre')

from keras import callbacks
#Allow model to stop training if val loss ever goes up from one epoch to another
earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)

r = model.fit(train_set, validation_data=test_set, epochs = 10, batch_size= 64, callbacks=[earlystopping])

model.save('./saved_models/model_newsave')

