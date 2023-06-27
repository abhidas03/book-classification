from keras.layers import Dense, Flatten
import keras.models
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator

import pandas as pd

vgg = VGG16(weights='vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape=(224, 224, 3), include_top=False, classes=30)
for layer in vgg.layers:
    layer.trainable = False

trainDataDf = pd.read_csv("input/bookcover30-labels-train.csv", header=None, encoding_errors='replace')
trainDataDfLess = trainDataDf.head(5000)
testDataDf = pd.read_csv("input/bookcover30-labels-test.csv", header=None, encoding_errors='replace')
testDataDfLess = testDataDf.head(500)


trainDataDfLess.iloc[:, 0] = 'input/224x224/' + trainDataDfLess.iloc[:, 0]
testDataDfLess.iloc[:, 0] = 'input/224x224/' + testDataDfLess.iloc[:, 0]
trainDataDfLess.columns = ['file', 'genre']
testDataDfLess.columns = ['file', 'genre']

print(trainDataDfLess.head())
print(testDataDfLess.head())

x = Flatten()(vgg.output)

prediction = Dense(30, activation='softmax')(x)

model = Model(inputs=vgg.input, outputs=prediction)

model.summary()

from keras import optimizers

adam = optimizers.legacy.Adam()


model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False)
test_datagen = ImageDataGenerator(rescale=1./255)


train_set = train_datagen.flow_from_dataframe(trainDataDfLess,
                                              target_size=(224, 224),
                                              batch_size= 64,
                                              class_mode='categorical',
                                              x_col='file',
                                              y_col='genre')

test_set = test_datagen.flow_from_dataframe(testDataDfLess,
                                            target_size=(224, 224),
                                            batch_size= 64,
                                            class_mode='categorical',
                                            x_col='file',
                                            y_col='genre')

from keras import callbacks
earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                        mode="min", patience=5,
                                        restore_best_weights=True)

r = model.fit(train_set, validation_data=test_set, epochs = 1, batch_size= 64, callbacks=[earlystopping])

model.save('./saved_models/model_firstsave')
model = keras.models.load_model('./saved_models/model_firstsave')
print(model.summary())