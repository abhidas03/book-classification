#TO BE USED WITH a saved model
import keras.models
import tensorflow as tf
import numpy as np
from collections import defaultdict

num_to_class = {0: 'Arts & Photography', 1: 'Biographies & Memoirs', 2:'Business & Money', 3: 'Calendars', 
                4: 'Children\'s Books',	5: 'Comics & Graphic Novels', 6:'Computers & Technology',  7: 'Cookbooks, Food & Wine',
                8: 'Crafts, Hobbies & Home', 9: 'Christian Books & Bibles', 10: 'Engineering & Transportation',  11: 'Health, Fitness & Dieting',
                12: 'History',  13: 'Humor & Entertainment', 14: 'Law',  15: 'Literature & Fiction',
                16: 'Medical Books', 17: 'Mystery, Thriller & Suspense', 18: 'Parenting & Relationships', 19: 'Politics & Social Sciences',
                20: 'Reference', 21: 'Religion & Spirituality', 22: 'Romance', 23: 'Science & Math',
                24: 'Science Fiction & Fantasy', 25: 'Self-Help', 26: 'Sports & Outdoors', 27: 'Teen & Young Adult',
                28: 'Test Preparation', 29: 'Travel' }

# for each category, will have a list of two numbers
# first number: # times a book of that category was guessed correctly
# second number: # times a book of that category was guessed incorrectly
amountCorrectForEachCategory = defaultdict(list)


model = keras.models.load_model("./saved_models/model_secondsave/")

def convertImage(filePath):
    image = tf.keras.utils.load_img(filePath)
    input_arr = tf.keras.utils.img_to_array(image)
    return input_arr

# open file
# for each line
# get prediction for image
# If prediction matches true category
# add one to first number of value for dictionary
# if wrong add one to second number
f = open("bookcover30-labels-test.txt")
from keras.applications.vgg16 import preprocess_input
for line in f:
    image, category = line.split()
    temp = image
    #----
    # if (int(category) != 4):
    #     continue
    #----
    image = np.array(convertImage('input/224x224/{0}'.format(image)))
    image = image.reshape(-1, 224, 224, 3)
    image = image.astype('float32')
    # preprocess_input(image)
    image = image/255

    y_prob = model.predict(image)
    y_classes = y_prob.argmax(axis=-1)[0]
    category = int(category)
    if (y_classes == category):
        amountCorrectForEachCategory.setdefault(category, [0, 0])[0] += 1
    else:
        amountCorrectForEachCategory.setdefault(category, [0, 0])[1] += 1
    #----
    # print('{0} Actual:{1} Prediction:{2}'.format(temp, num_to_class[category], num_to_class[y_classes]))
    #----

for key in amountCorrectForEachCategory:
    key = int(key)
    print('Amount correct for {}: {}/{}'.format(num_to_class[key], amountCorrectForEachCategory[key][0], amountCorrectForEachCategory[key][1] + amountCorrectForEachCategory[key][0]))

#<<<
# y_prob = model.predict(image)
# y_classes = y_prob.argmax(axis=-1)
# print(num_to_class[y_classes[0]])
#<<<