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
        regTheta = np.subtract(theta, 0)
        regTheta[0] = 0
        term3 = np.transpose(regTheta).dot(regTheta)
        term3 = lambd*term3/2
        sum = term1 + term2 - term3
        cost = sum/numrow
        return -cost
        
    def Gradient(self, X, y, theta, lambd):
        numrow = X.shape[0]         #number of rows
        ones = np.ones(numrow)
        X = np.c_[ones, X]          #now we have correct X matrix 
        hypthesis = X.dot(theta)
        hypthesis = self.Sigmoid(hypthesis)
        grad_term1_sub = np.subtract(hypthesis, y)
        grad_term1_sub = np.transpose(grad_term1_sub).dot(X)
        grad_term1 = np.transpose(grad_term1_sub)
        reg_theta = regTheta = np.subtract(theta, 0)
        reg_theta[0] = 0
        grad_term2 = reg_theta*lambd
        sum = grad_term1  + grad_term2
        grad = sum/numrow
        return grad

    def printDataset(self):
        print(self.dataset)
        print(sorted(self.dataset.keys()))
        

#    def Train():
        
        