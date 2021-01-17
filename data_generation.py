# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:02:16 2018

@author: Colm Keyes
"""
import numpy as np
from PIL import ImageGrab
import cv2
import time
#from directkeys import ReleaseKey, PressKey, W, A, S, D
from getkeys import key_check
import os

def keys_to_output(keys):
    #        [A,W,D]
    output = [0,0,0] # This is Boolean algebra as numbers will only be 0 or 1
        
    if 'A' in keys:
        output[0] = 1
    elif 'D' in keys:
        output[2] = 1
    else:# 'W' in keys:
        output[1] = 1
    
    return output

file_name = 'training_data.npy'
if os.path.isfile(file_name):
    print('File exists, loading previous data.')
    #for appending to the training data
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh.')
    training_data = []

def main(): 
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)       
    last_time = time.time()
    while True:
        screen = np.array(ImageGrab.grab(bbox=(0,40,800,640)))
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        screen = cv2.resize(screen,(80,64))
        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([screen,output])
        if len(training_data) % 500 ==0:
            print(len(training_data))
            np.save(file_name, training_data)
        
main()