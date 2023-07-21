#TO BE USED WITH a saved model
import keras.models
import tensorflow as tf
import numpy as np
from keras.applications.vgg16 import preprocess_input
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

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
    
num_to_class = {0: 'Arts & Photography', 1: 'Biographies & Memoirs', 10:'Business & Money', 11: 'Calendars', 
                12: 'Children\'s Books', 13: 'Comics & Graphic Novels', 14:'Computers & Technology',  15: 'Cookbooks, Food & Wine',
                16: 'Crafts, Hobbies & Home', 17: 'Christian Books & Bibles', 18: 'Engineering & Transportation',  19: 'Health, Fitness & Dieting',
                2: 'History',  20: 'Humor & Entertainment', 21: 'Law',  22: 'Literature & Fiction',
                23: 'Medical Books', 24: 'Mystery, Thriller & Suspense', 25: 'Parenting & Relationships', 26: 'Politics & Social Sciences',
                27: 'Reference', 28: 'Religion & Spirituality', 29: 'Romance', 3: 'Science & Math',
                4: 'Science Fiction & Fantasy', 5: 'Self-Help', 6: 'Sports & Outdoors', 7: 'Teen & Young Adult',
                8: 'Test Preparation', 9: 'Travel' }

wrong_to_right = {0: 0, 1: 1, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18, 
               11: 19, 12: 2, 13: 20, 14: 21, 15: 22, 16: 23, 17: 24, 18: 25, 19: 26, 20: 27, 
               21: 28, 22: 29, 23: 3, 24: 4, 25: 5, 26: 6, 27: 7, 28: 8, 29: 9}

bad_to_good = {0: 0, 1: 1, 2: 10, 3: 11, 4: 12, 5: 13, 6: 14, 7: 15, 8: 16, 9: 17, 10: 18, 
               11: 19, 12: 2, 13: 20, 14: 21, 15: 22, 16: 23, 17: 24, 18: 25, 19: 26, 20: 27, 
               21: 28, 22: 29, 23: 3, 24: 4, 25: 5, 26: 6, 27: 7, 28: 8, 29: 9}
#load saved model
model = keras.models.load_model("./saved_models/model_newsave/")

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
    image = preprocess_input(image)


    y_prob = model.predict(image)
    prediction = y_prob.argmax(axis=-1)[0]
    prediction = bad_to_good[prediction]
    y_pred.append(prediction)


#Create Confusion Matrix
y_labels = np.array(num_to_class.values())
y_true = np.array(y_true)
y_pred = np.array(y_pred)
cm = confusion_matrix(y_true=y_true, y_pred=y_pred, normalize='true')
print(cm)
np.savetxt('goodNums.txt', cm)

'''
EVERYTHING ABOVE IN SSH - THEN COPY TXT FILE TO LAPTOP, 
COMMENT OUT CODE ABOVE, RUN JUST THE CODE COMMENTED BELOW
'''

# cm = np.loadtxt('confusion.txt')

# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_labels)
# disp.plot(include_values=False, xticks_rotation='vertical')
# plt.show()
