# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 07:58:00 2019
Description: Main class that uses both Trainer and Predicter class for 
debugging.
@author: OrduLou
"""
import numpy as np
from Trainer import Trainer
import scipy.io as sio
class main:
    
    path = 'sampleDatabase.mat'
#    dataset = sio.loadmat(path)
#    X = dataset['X']
#    y = dataset['y']
    a =((np.arange(15) +1).reshape((3,5)))/10
    a = np.transpose(a)
#    ones= np.ones(5)
#    a = np.c_[ones, a]
    y = np.array([[1],[0], [1],[0] , [1]])
    lambd = 3
    theta = np.array([[-2], [-1], [1], [2]])
    trainer = Trainer(path)
    print(trainer.costFunction(a, y, theta, lambd))
    print('\n---------------------------\n')
    print(trainer.Gradient(a, y, theta, lambd))
#    print(np.sum(a, 0))
#    print(np.multiply(1-b,np.log(1-a)))
#    print(a)
#    
#     print(trainer.costFunction(X,y))
#    trainer.loadDataset()
#    trainer.costFunction()
    