# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 07:58:00 2019
Description: Main class that uses both Trainer and Predicter class for 
debugging.
@author: OrduLou
"""
import numpy as np
from Trainer import Trainer
from Predicter import Predicter 
import scipy.io as sio
class main:
    
    path = 'sampleDatabase.mat'
    ### these comments were used to test function of Trainer class.
#    dataset = sio.loadmat(path)
#    X = dataset['X']
#    y = dataset['y']
#    a =((np.arange(15) +1).reshape((3,5)))/10
#    a = np.transpose(a)
#    ones= np.ones(5)
#    a = np.c_[ones, a]
#    y =     np.array([[1], [2], [1], [2], [1]])
#    lambd = 3
#    theta = np.array([[1], [3], [1], [5], [6]])
#    deneme = np.array([1])
#    print(theta[2][0])
#    print(np.vstack([deneme, y]))
#    trainer = Trainer(path)
#    trainer.loadDataset()
#    trainer.Train()
#    trainer.Predict()
    i = 28
    dataset = sio.loadmat(path)
    X = dataset['X']
    y = dataset['y']
    predicter = Predicter()
    predicter.loadModel()
    sample = X[i]
#    print(sample.shape)
#    sample = np.transpose(sample)
#    print(sample.shape)
    predicter.predict(sample)
    print(y[i])
    
    