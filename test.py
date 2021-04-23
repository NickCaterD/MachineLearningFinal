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
import DefineModel


input_shape = (512,512,3)
classes = 11
model = DefineModel.createModel(input_shape,classes)
model.load_weights("model_weights.h5")

img_path = "data/..."
img = cv2.imread(img_path)

# class probability distribution
softmax_outputs = model.predict(img)

# most likely class
pred_label = softmax_outputs.argmax()

names = ['','','','','','','','','','','']