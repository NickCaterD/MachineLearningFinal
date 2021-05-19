import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
import DefineModel_OCTnet
import DefineModel
import time
import re
import matplotlib.pyplot as plt
import data_aug
from data_aug import master_augment

# Define variables
input_shape = (227, 227,1)
classes = 11

start = time.time()
# Import data from pretrain to arrays somehow
X = []
Y = []

path = '../ODIR-5K/ODIR-5K/Training Images'
path_preproc = '../preprocessed_images'

count = 0
for file_name in os.listdir(path):
    if file_name == '.ipynb_checkpoints':
        break
    im = cv2.imread(os.path.join(path_preproc, file_name))
    if np.shape(im) != (512,512,3):
        continue
    split = re.split(r'[.,]',file_name)
    label_name = split[0] + '.txt'
    label = np.loadtxt(os.path.join('labels', label_name))
    count+=1
    if (count%500 == 0):
        print(count)
#     if count == 3000:
#         break

    #lines = text_file.read().split(',')
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = cv2.resize(im,(227,227))
    #im = im[:, :, np.newaxis]
    augmented_images = data_aug.master_augment(im)
    for img in augmented_images:
        X.append(img/255)
        Y.append(label)
           
    
# convert to numpy array for training
X = np.array(X, dtype = np.float32)
Y = np.array(Y, dtype = np.float32)

X = X.reshape(len(X),227,227,1)


print(np.shape(X), np.shape(X[0]))
print(np.shape(Y))
model = DefineModel_OCTnet.createModel(input_shape,classes)
print(model.summary())



# compile model, might need changes to loss and optimizer

model.compile(optimizer=optimizers.Adam(epsilon = 0.1), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

#data augmentation

##
# sgd = optimizers.SGD(lr = 0.001, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd,
#               loss='categorical_crossentropy',
#               metrics=['accuracy'])


print('-TRAINING----------------------------')
# print('Input shape:', X.shape)
# print('Number Images: ', X.shape[0])

# Whatever we want here
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, train_size=0.8,test_size=0.2, shuffle= True)

# Def change epochs and batch size
# 
history = model.fit(x=x_train,y=y_train, epochs=10, batch_size = 512, validation_data=(x_valid, y_valid))

# serialize weights to HDF5
model.save_weights("model_weights_grayscale.h5")

# Loss Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['loss'],'r',linewidth=3.0)
plt.plot(history.history['val_loss'],'b',linewidth=3.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Loss',fontsize=16)
plt.title('Loss Curves',fontsize=16)
plt.savefig('loss_curve_grayscale2.jpg')
plt.show()

# Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(history.history['accuracy'],'r',linewidth=3.0)
plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)
plt.savefig('training_curve_grayscale2.jpg')
plt.show()

