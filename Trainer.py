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
        
    def Sigmoid(self, h):
        return 1 / (1 + np.exp(-h))
        
    def costFunction(X, y, theta):
        numrow = X.shape[0]         #number of rows
        ones = np.ones(numrow)
        X = np.c_[ones, X]          #now we have correct X matrix
        length = X.shape[1]         #number of columns
        hypthesis = X.dot(theta)
        hypthesis = Sigmoid(hypthesis)
        term1 = np.multiply(y, log(hypthesis))
        term2 = np.multiply(1-y , log(1-hypthesis))
        term1 = np.sum(term1)
        term2 = np.sum(term2)
        return term1 + term2
        
#    def Gradient():
        

    def printDataset(self):
        print(self.dataset)
        print(sorted(self.dataset.keys()))
    
        
#    def Train():
        
        