import keras.models
import tensorflow as tf
import numpy as np
num_to_class = {0: 'Arts & Photography', 1: 'Biographies & Memoirs', 2:'Business & Money', 3: 'Calendars', 
                4: 'Children\'s Books',	5: 'Comics & Graphic Novels', 6:'Computers & Technology',  7: 'Cookbooks, Food & Wine',
                8: 'Crafts, Hobbies & Home', 9: 'Christian Books & Bibles', 10: 'Engineering & Transportation',  11: 'Health, Fitness & Dieting',
                12: 'History',  13: 'Humor & Entertainment', 14: 'Law',  15: 'Literature & Fiction',
                16: 'Medical Books', 17: 'Mystery, Thriller & Suspense', 18: 'Parenting & Relationships', 19: 'Politics & Social Sciences',
                20: 'Reference', 21: 'Religion & Spirituality', 22: 'Romance', 23: 'Science & Math',
                24: 'Science Fiction & Fantasy', 25: 'Self-Help', 26: 'Sports & Outdoors', 27: 'Teen & Young Adult',
                28: 'Test Preparation', 29: 'Travel' }

model = keras.models.load_model("./saved_models/model_firstsave/")

def convertImage(filePath):
    image = tf.keras.utils.load_img(filePath)
    input_arr = tf.keras.utils.img_to_array(image)
    return input_arr

image = np.array(convertImage('input/224x224/1579003907.jpg'))

image = image.reshape(-1, 224, 224, 3)
image = image.astype('float32')

y_prob = model.predict(image)
y_classes = y_prob.argmax(axis=-1)
print(num_to_class[y_classes[0]])