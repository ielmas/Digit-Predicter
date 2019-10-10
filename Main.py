# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 07:58:00 2019
Description: Main class that uses both Trainer and Predicter class for 
debugging.
@author: OrduLou
"""
import numpy as np
from Trainer import Trainer

class main:
    
    path = 'sampleDatabase.mat'
    dataset = sio.loadmat(path)
    X = dataset.contents['X']
    y = dataset.contents['y']
    a =(np.arange(9) +1).reshape((3,3))
    c = np.array([[-2], [-2], [-2]])
    b = np.array([[2], [2], [2]])
    print(np.sum(a, 0))
#    print(np.multiply(1-b,np.log(1-a)))
#    print(a)
    trainer = Trainer(path)
#    trainer.loadDataset()
#    trainer.costFunction()
    