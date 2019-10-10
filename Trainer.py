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
        
    def costFunction(self, X, y, theta, lambd):
        numrow = X.shape[0]         #number of rows
        ones = np.ones(numrow)
        X = np.c_[ones, X]          #now we have correct X matrix 
        hypthesis = X.dot(theta)
        hypthesis = self.Sigmoid(hypthesis)
        term1 = np.multiply(y, np.log(hypthesis))
        term2 = np.multiply(1-y , np.log(1-hypthesis))
        term1 = np.sum(term1)
        term2 = np.sum(term2)
        reg_theta = theta
        reg_theta[0] = 0
        term3 = np.transpose(reg_theta).dot(reg_theta)
        term3 = lambd*term3/2
        sum = term1 + term2 - term3
#        cost = (-1/numrow)(term1 + term2 - term3)
        return sum
        
#    def Gradient():
        

    def printDataset(self):
        print(self.dataset)
        print(sorted(self.dataset.keys()))
    
        
#    def Train():
        
        