# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 07:41:15 2019

@author: OrduLou
"""

import numpy as np
import scipy.io as sio
import scipy.optimize as opt
class Trainer:
    def __init__(self, path):
        self.path = path
    
    def loadDataset(self):
        self.dataset = sio.loadmat(self.path)
        
    def Sigmoid(self, h):
        return 1 / (1 + np.exp(-h))
        
    def costFunction(self, theta, X, y, lambd):
        numrow = X.shape[0]         #number of rows
#        ones = np.ones(numrow)
#        X = np.c_[ones, X]          #now we have correct X matrix 
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
        
    def Gradient(self, theta, X, y, lambd):
        numrow = X.shape[0]         #number of rows
#        ones = np.ones(numrow)
#        X = np.c_[ones, X]          #now we have correct X matrix 
        hypthesis = X.dot(theta)
        hypthesis = self.Sigmoid(hypthesis)
        grad_term1_sub = hypthesis- y
        grad_term1_sub = np.transpose(grad_term1_sub).dot(X)
        grad_term1 = np.transpose(grad_term1_sub)
        reg_theta = np.subtract(theta, 0)
        reg_theta[0] = 0
        grad_term2 = reg_theta*lambd
        sum = grad_term1  + grad_term2
        grad = sum/numrow
        return grad

    def printDataset(self):
        print(self.dataset)
        print(sorted(self.dataset.keys()))
        

    def Train(self):
        X = self.dataset['X']
        numrow = X.shape[0]         #number of rows
        ones = np.ones(numrow)
        X = np.c_[ones, X]          #now we have correct X matrix 
        y = self.dataset['y']
        features = X.shape[1]
        numberLabels = 10 # since there are 10 digits
        self.all_thetas = np.zeros((numberLabels, features))
        lambd = 0.1
        for x in range(10):
            print('AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA')
            init_theta = np.zeros(features)
            condtn = x
            if (x == 0):
                condtn = 10
            output = opt.fmin_tnc(func = self.costFunction, x0 = init_theta.flatten(), fprime = self.Gradient, \
                         args = (X, (y == condtn).flatten(), lambd))
            self.all_thetas[x] = output[0]
            print(self.all_thetas[x])
        np.savetxt("thetaValues.csv", self.all_thetas, delimiter=",")
    def Predict(self):
        X = self.dataset['X']
        numrow = X.shape[0]         #number of rows
        ones = np.ones(numrow)
        X = np.c_[ones, X]          #now we have correct X matrix 
        y = self.dataset['y']
        theta_tra = np.transpose(self.all_thetas)
        probs = X.dot(theta_tra)
        indices = np.transpose(np.argmax(probs, 1))
        real_indices = np.where(indices == 0, 10, indices)
        value = 0
        for x in range(5000):
            if(real_indices[x] == y[x]):
                value = value+ 1
        print(value)
#        print(real_indices)
#        print(y.shape)
#        total = np.equal(y, real_indices)
#        print(total.shape)
        