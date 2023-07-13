#TO BE USED WITH a saved model
import keras.models
import tensorflow as tf
import numpy as np
from keras.applications.vgg16 import preprocess_input
import matplotlib
matplotlib.use('GTK3Agg')
import matplotlib.pyplot as plt

# PLAN FOR VISUALIZING DATA
# To get true labels array and predictions array
    # Open file
    # For each line 
        # Split line into image path and true label
        # Add true label to true label array
        # convert each image into correct format
        # get Prediction for image 
        # Add prediction to predictions Array
    #Feed true label array and prediction array into sklearn metrics confusion matrix
    #Profit
    
num_to_class = {0: 'Arts & Photography', 1: 'Biographies & Memoirs', 2:'Business & Money', 3: 'Calendars', 
                4: 'Children\'s Books',	5: 'Comics & Graphic Novels', 6:'Computers & Technology',  7: 'Cookbooks, Food & Wine',
                8: 'Crafts, Hobbies & Home', 9: 'Christian Books & Bibles', 10: 'Engineering & Transportation',  11: 'Health, Fitness & Dieting',
                12: 'History',  13: 'Humor & Entertainment', 14: 'Law',  15: 'Literature & Fiction',
                16: 'Medical Books', 17: 'Mystery, Thriller & Suspense', 18: 'Parenting & Relationships', 19: 'Politics & Social Sciences',
                20: 'Reference', 21: 'Religion & Spirituality', 22: 'Romance', 23: 'Science & Math',
                24: 'Science Fiction & Fantasy', 25: 'Self-Help', 26: 'Sports & Outdoors', 27: 'Teen & Young Adult',
                28: 'Test Preparation', 29: 'Travel' }

model = keras.models.load_model("./saved_models/model_secondsave/")

def convertImage(filePath):
    image = tf.keras.utils.load_img(filePath)
    input_arr = tf.keras.utils.img_to_array(image)
    return input_arr


f = open("bookcover30-labels-test.txt")
y_true = []
y_pred = []
for line in f:
    #Split line into image path and true label
    image, trueLabel = line.split()

    #Append to true labels
    y_true.append(int(trueLabel))

    #Convert images
    image = np.array(convertImage('input/224x224/{0}'.format(image)))
    image = image.reshape(-1, 224, 224, 3)
    image = image.astype('float32')
    preprocess_input(image)

    y_prob = model.predict(image)
    y_pred.append(y_prob.argmax(axis=-1)[0])

y_labels = list(num_to_class.values())
print('pred')
print(y_pred)
print('true')
print(y_true)
print('labels')
print(y_labels)

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
print(cm)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_labels)
disp.plot()
plt.show()
