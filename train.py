# Train model on ocular disease images

import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import DefineModel

# Define variables
input_shape = (512, 512, 3)
classes = 11

# Import data from pretrain to arrays somehow
X = []
Y = []

path = '/MachineLearningFinal/ODIR\-5K/ODIR\-5K/Training\sImages'

for file_name in os.listdir(path):
    im = cv2.imread(os.path.join(path, file_name))
    label = np.loadtext(os.path.join('labels', file_name))
    X.append(im)
    Y.append(label)    
# convert to numpy array for training
X = np.array(X)
Y = np.array(Y)

# make sure it worked
#cv2.imshow('1st sample', X[10])
#print(Y[0])

model = DefineModel.createModel(input_shape,classes)
print(model.summary())

# compile model, might need changes to loss and optimizer

model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
"""
sgd = optimizers.SGD(lr = 0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
"""

print('-TRAINING----------------------------')
print('Input shape:', X.shape)
print('Number Images: ', X.shape[0])

# Whatever we want here
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.2, shuffle= True)

# Def change epochs and batch size
history = model.fit(x=x_train,y=y_train, epochs=10, batch_size=64, validation_data=(x_valid, y_valid))

# serialize weights to HDF5
model.save_weights("model_weights.h5")

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.show()

# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.show()
