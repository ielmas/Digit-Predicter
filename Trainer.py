# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 07:41:15 2019

@author: OrduLou
"""

import numpy as np
import scipy.io as sio

class Trainer:
    def __init__(self, path):
        self.path = path
    
    def loadDataset(self):
        self.dataset = sio.loadmat(self.path)
        

    def costFunction(self):
        X = self.dataset['X']
        y = self.dataset['y']
        numrow = X.shape[0]         #number of rows
        ones = np.ones(numrow)
        X = np.c_[ones, X]          #now we have correct X matrix
        length = X.shape[1]         #number of columns
        init_theta = np.zeros(length) 
        
        

    def printDataset(self):
        print(self.dataset)
        print(sorted(self.dataset.keys()))