import numpy as np
import keras
from skimage.io import imread
from skimage.transform import resize

class DataGenerator(keras.utils.Sequence):
    def __init__(self, image_filenames, labels, batch_size): 
        self.image_filenames = image_filenames
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return(np.ceil(len(self.image_filenames) / float(self.batch_size))).astype(int)
    
    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size: (idx+1) * self.batch_size]
        batch_y = self.labels[idx * self.batch_size : (idx+1) * self.batch_size]

        return np.array([
            resize(imread('input/224x224/' + str(file_name)), (80, 80, 3))
                for file_name in batch_x])/255.0, np.array(batch_y)
        