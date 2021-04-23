# Define CNN archicecture

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import optimizers

def createModel(input_shape,classes):
    
    # Creating a Sequential Model and adding the layers
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5,5), strides = (1,1), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(64, kernel_size=(3,3), strides = (1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(128, kernel_size=(3,3), strides = (1,1), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3,3), strides = (1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    
    model.add(Conv2D(256, kernel_size=(3,3), strides = (1,1), activation='relu'))
    model.add(Conv2D(256, kernel_size=(3,3), strides = (1,1), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    # Flattening the 2D arrays for fully connected layers
    model.add(Flatten()) 
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(classes,activation='softmax'))

    return model
