import cv2
import os
import tensorflow as tf
from tensorflow import keras
from utils import normalize
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam

class EmotionDetectionModel():
    def __init__(self):
        self.width = 256
        self.height = 256
        self.width_crop = 48
        self.height_crop = 48
        self.training_folder = "./training_dataset/"
    
    def create_model(self):
        self.model = tf.keras.models.Sequential([
            keras.layers.InputLayer(input_shape=(self.width_crop * self.height_crop,)),
            keras.layers.Dense(32, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(64, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(128, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1024, activation = 'relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            keras.layers.Dropout(0.1),
            #0 - sad
            #1 - neutral
            #2 - happy
            #3 - fear
            #4 - surprise
            #5 - anger
            #6 - disgust
            tf.keras.layers.Dense(7, activation = 'softmax')
            ])
        
        self.model.compile(optimizer = 'adam', 
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics = ['accuracy'])
        
        return self.model
    
    def gather_data(self):
        x_load_images = []
        y_load_images = []
        for filename in os.listdir(self.training_folder):
            #Read one image from folder
            img = cv2.imread(os.path.join(self.training_folder,filename))
            if img is not None:
                #Normalize all images to the same dimensions
                img = normalize(img, self.width, self.height)
                
                #Detect Face
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(img, 1.1, 1)
                for (x, y, w, h) in faces:
                    #Crop the face
                    crop_img = img[y:y+h, x:x+w]
                    #Draw Rectangle for test
                    #cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    
                    #Resize the faces only and normalize the pixels to be between [0,1]
                    crop_img = cv2.resize(crop_img, (self.width_crop, self.height_crop), interpolation = cv2.INTER_AREA)
                    crop_img = crop_img / 255
                    x_load_images.append(crop_img)
                    
                    #Create responses based on filename
                    if "sad" in filename:
                        y_load_images.append(0)
                    elif "neutral" in filename:
                        y_load_images.append(1)
                    elif "happy" in filename:
                        y_load_images.append(2)
                    elif "fear" in filename:
                        y_load_images.append(3)
                    elif "surprise" in filename:
                        y_load_images.append(4)
                    elif "anger" in filename:
                        y_load_images.append(5)
                    elif "disgust" in filename:
                        y_load_images.append(6)
                    #Display test
                    #cv2.imshow('crop', crop_img)
                    #cv2.imshow('no crop', img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    
        return (x_load_images, y_load_images)
    
    def train(self):
        #Get the training data
        x_load_images = []
        y_load_images = []
        
        x_load_images, y_load_images = self.gather_data()
        
        #Training
        
        #Transform to np array
        x_faces = np.array(x_load_images, dtype=(float)).reshape(-1, self.width_crop * self.height_crop)
        y_faces = np.array(y_load_images, dtype=(int))
        
        #Split into trainable data
        x_train, x_test, y_train, y_test = train_test_split(x_faces, y_faces, 
                                                            test_size = 0.2, shuffle=(True))
        
        tf.keras.utils.normalize(x_train, order=2)
        tf.keras.utils.normalize(x_test, order=2)
        
        #Create the model and train
        self.model = self.create_model()
        
        history = self.model.fit(x_train, y_train, epochs=210, batch_size = 15)
        
        loss, accuracy = self.model.evaluate(x_test, y_test, verbose=2)
        
        print("Accuracy :" + str(accuracy))

        plt.plot(history.history['loss'])
        plt.plot(history.history['accuracy'])
        plt.legend(['loss', 'accuracy'], loc='upper right')
        plt.show()