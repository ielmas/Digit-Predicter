# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:00:24 2019

@author: OrduLou
"""
import pandas as pd
import numpy as np

class Predicter:
    def Sigmoid(self, h):
        return 1 / (1 + np.exp(-h))
    def loadModel(self):
        self.thetas = pd.read_csv('thetaValues.csv', sep=',',header=None)
    def predict(self, X):
#        print(X.shape)
#        numrow = X.shape[0]         #number of rows
#        print(numrow)
#        ones = np.ones(numrow)
#        X = np.c_[ones, X]          #now we have correct X matrix 
#        one = np.array([1])
        newX = np.zeros((1, 401))
        newX[0][0] = 1
        for i in range(400):
            print(i)
            if(i != 0):
                newX[0][i+1] = X[i]
#        X = np.vstack([one, X])
        probs = newX.dot(np.transpose(self.thetas))
        probs = self.Sigmoid(probs)
        predicted = np.argmax(probs, 1)
        print(predicted)