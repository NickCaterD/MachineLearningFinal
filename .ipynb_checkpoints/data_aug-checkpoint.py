import cv2
import numpy as np

def lighting(img):
    # can do multiple magnitudes
    #mags = [20,40,60,80]
    mags = [20,40,60,80]
    light_list = []
    for mag in mags:
        bright = np.ones(img.shape, dtype="uint8")*mag
        increase = cv2.add(img,bright)
        decrease = cv2.subtract(img,bright)
        light_list.append(increase)
        light_list.append(decrease)
    return light_list

def flip(img):
    mirror_y = cv2.flip(img,1)
    return mirror_y

def translate(img):
    # Store height and width of the image
    height, width = img.shape[:2]
    shift_height, shift_width = int(height/50), int(width/50)

    # Define shift matrices
    T1 = np.float32([[1, 0, shift_width], [0, 1, shift_height]])
    T2 = np.float32([[1, 0, -shift_width], [0, 1, -shift_height]])
    T3 = np.float32([[1, 0, shift_width], [0, 1, -shift_height]])
    T4 = np.float32([[1, 0, -shift_width], [0, 1, shift_height]])
  
    # Use warpAffine to transform the image using the matrix, T
    s1 = cv2.warpAffine(img, T1, (width, height))
    s2 = cv2.warpAffine(img, T2, (width, height))
    s3 = cv2.warpAffine(img, T3, (width, height))
    s4 = cv2.warpAffine(img, T4, (width, height))
    
    return [s1,s2,s3,s4]

def master_augment(img):
    all_imgs = []
    lighting_imgs = lighting(img)
    for img in lighting_imgs:
        all_imgs.append(img)
        
    flipped_img = flip(img)
    all_imgs.append(flipped_img)
    
    translated_imgs = translate(img)
    for img in translated_imgs:
        all_imgs.append(img)
    all_imgs.append(img)
    
    return all_imgs