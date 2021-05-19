# this is test
import cv2
import numpy as np
"""
from skimage import measure
from skimage.color import rgb2lab
"""
import os
# import random as rand
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
import DefineModel_OCTnet
from tensorflow.keras import optimizers


input_shape = (227,227,1)
classes = 11
model = DefineModel_OCTnet.createModel(input_shape,classes)
model.load_weights("model_weights_grayscale.h5")

# Prediction for one sample
###########################################################################
"""
img_path = "../ODIR-5K/ODIR-5K/Testing Images/1000_left.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (227,227))

img_array = np.array([img])
img_array = img_array.reshape(1,227,227,1)

# class probability distribution
softmax_outputs = model.predict(img_array)
print(softmax_outputs)

# most likely class
pred_label = softmax_outputs.argmax()
print(pred_label)

names = ['normal fundus',\
            'moderate non proliferative retinopathy',\
            'mild nonproliferative retinopathy',
           'glaucoma',\
            'pathological myopia',\
            'dry age-related macular degeneration',\
            'hypertensive retinopathy',\
            'epiretinal membrane',\
            'drusen',\
            'macular epiretinal membrane',
            'other']

print("pred = ", names[pred_label])
"""
path = '../ODIR-5K/ODIR-5K/Testing Images'
path_preproc = '../preprocessed_images'
# Evaluation for test set
#######################################################################
test_x = []
test_y = []
count = 0
for file_name in os.listdir(path):
    if file_name == '.ipynb_checkpoints':
        pass
    im = cv2.imread(os.path.join(path, file_name))
    if np.shape(im) != (512,512,3):
         continue
    split = re.split(r'[.,]',file_name)
    label_name = split[0] + '.txt'
    label = np.loadtxt(os.path.join('labels', label_name))
    im = cv2.resize(im,(227,227))
    print(np.shape(im))
    im = im/255
    testing_set.append(im)
    
    count+=1
    if count == 2:
        break
    test_y.append(label) 

test_x = np.array(test_x, dtype = np.float32)
test_y = np.array(test_y, dtype = np.float32)

test_x = test_x.reshape(len(test_x),227,227,1)
    
model.compile(optimizer=optimizers.Adam(epsilon = 0.1), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
results = model.evaluate(test_x, test_y, batch_size=128)