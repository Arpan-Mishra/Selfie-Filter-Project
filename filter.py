import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd
import os

# Loading our model
keypoint_detector = load_model('Keypoint_CNN.h5')

# loading our face detector (HAAR CASCADE CLASSIFIER)
face_cascade = cv2.CascadeClassifier('/Cascade%20Classifier/haarcascade_frontalface_default.xml')

# our filters
filters = ['Images/shades.png/','Images/pipe.png/']
# keypoint names
keypoint_names = np.load('Data/Keypoints_name.npy',allow_pickle = True)

# captureing the video
cap = cv2.VideoCapture(0)

# streaming 
while True:
    _,frame = cap.read()
    frame  = cv2.flip(frame,1)
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_copy = np.copy(frame) # we`ll be displayimg 2 frames 
    
    # converting to grayscale so that model can detect keypoints
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     
    # detecting face
    faces = face_cascade.detectMultiScale(gray_frame,1.25,6)
    
    # looping over the faces
    for (x,y,w,h) in faces:
        
        #selecting the face region for keypoint detector
        face_gray = gray_frame[y:y+h,x:x+w]
        face_color = frame_copy[y:y+h,x:x+w] 
        # resizing and scaling
        face_gray_scaled = face_gray/255.
        shape_orig = face_gray.shape
        face_gray_resized = cv2.resize(face_gray_scaled,(96,96),interpolation = cv2.INTER_AREA)
        face_resized_copy = face_gray_resized.copy()
        face_gray_resized = face_gray_resized.reshape(1,96,96,1)
        
        # predicting our keypoints
        keypoints = keypoint_detector.predict(face_gray_resized)
        
        # converting into a more usable format
        keypoints_df = pd.DataFrame(keypoints,columns = keypoint_names)
        
        # SHADES FILTER
        
        # loading the filter
        shades = cv2.imread(filters[0],cv2.IMREAD_UNCHANGED)
        
        # defining our filter space 
        filter_width = int(1.2*abs(keypoints_df['right_eyebrow_outer_end_x'] - keypoints_df['left_eyebrow_outer_end_x']))
        filter_height =  int((abs(keypoints_df['nose_tip_y'] - keypoints_df['right_eyebrow_outer_end_y'])))
        filter_resized = cv2.resize(shades, (filter_width,filter_height),interpolation = cv2.INTER_CUBIC)
        
        
        # resizing face so that filter can be mapped
        face_color_resize = cv2.resize(face_color,(96,96),interpolation = cv2.INTER_AREA)
        face_color_resize_copy = face_color_resize.copy()
        
        # mapping to face
        mask = filter_resized[:,:,:3] != 0
        face_color_resize[int(keypoints_df['right_eyebrow_outer_end_y']):int(keypoints_df['right_eyebrow_outer_end_y'])+filter_height,
                          int(keypoints_df['right_eyebrow_outer_end_x']):int(keypoints_df['right_eyebrow_outer_end_x']) + filter_width][mask] = filter_resized[:,:,:3][mask]
        
        # mapping to frame
        frame[y:y+h, x:x+w] = cv2.resize(face_color_resize, shape_orig, interpolation = cv2.INTER_CUBIC)
        
        
        # PIPE FILTER
        pipe = cv2.imread(filters[1],cv2.IMREAD_UNCHANGED)
        pipe = cv2.cvtColor(pipe,cv2.COLOR_BGRA2BGR)
        pipe_height = abs(int(1.3*int(keypoints_df['mouth_center_top_lip_y']) - int(keypoints_df['mouth_center_bottom_lip_y'])))
        pipe_width = int(1.3*int(80 - int(keypoints_df['mouth_center_top_lip_x']) ))
        pipe_resize = cv2.resize(pipe,(pipe_width,pipe_height),interpolation = cv2.INTER_AREA)
        
        # mapping to face
        mask3 = pipe_resize[:,:,:3] != 0
        face_color_resize[int(keypoints_df['mouth_center_top_lip_y']-3):int(keypoints_df['mouth_center_top_lip_y']-3)+pipe_height,
                          int(keypoints_df['mouth_center_top_lip_x']):int(keypoints_df['mouth_center_top_lip_x'])+pipe_width][mask3] = pipe_resize[:,:,:3][mask3]
        
        
        # mapping to frame
        frame[y:y+h, x:x+w] = cv2.resize(face_color_resize, shape_orig, interpolation = cv2.INTER_CUBIC)

        
        # plotting the keypoints on the face
        for i in range(1,31,2): # 15 keypoints (x,y)
            key_T = keypoints.T
            cv2.circle(face_color_resize_copy,(key_T[i-1],key_T[i]),1,(0,0,255),1)
        frame_copy[y:y+h,x:x+w] = cv2.resize(face_color_resize_copy,shape_orig,interpolation = cv2.INTER_CUBIC)
        
        cv2.imshow('Keypoint Detector',frame_copy)
        cv2.imshow('Filter',frame)
    k = cv2.waitKey(0)
    if k == 27:
        break
       
cv2.destroyAllWindows()
cap.release()  

