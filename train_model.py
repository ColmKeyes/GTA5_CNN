# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:16:15 2018

@author: Colm Keyes
"""

import numpy as np
from alexnet import alexnet
import tensorflow as tf
WIDTH = 80
HEIGHT = 64
LR = 1e-3
EPOCHS = 8
MODEL_NAME = 'pygta5-car-{}-{}-{}-epochs.model'.format(LR, 'alexnetv2' , EPOCHS)

model = alexnet(WIDTH, HEIGHT, LR)

train_data = np.load('training_data_v2.npy')

train = train_data[:500]
test = train_data[-500:]

X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
#not eactly sure why you don't need the np.array here
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 1)
test_y = [i[1] for i in test]

model.fit({'input': X}, {'targets': Y}, n_epoch=EPOCHS, 
          validation_set = ({'input': test_x}, {'targets': test_y}), snapshot_step=500 ,
          show_metric= True, run_id = MODEL_NAME)

#for testing in windows, tensorboard --logdir=foo:C:/Users/Brian...... going to the log file 
#from the alexnet model at the end of alexnet


model.save(MODEL_NAME)







