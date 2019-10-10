# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:00:24 2019

@author: OrduLou
"""
import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import image
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
            if(i != 0):
                newX[0][i+1] = X[i]
        print(newX)
#        X = np.vstack([one, X])
        probs = newX.dot(np.transpose(self.thetas))
        probs = self.Sigmoid(probs)
        print(probs)
        predicted = np.argmax(probs, 1)
        print(predicted)
        
    def loadImage(self):
        img = Image.open('sample.jpg')
        grayImg = img.convert(mode='L')
        grayImg.save('sampleGray.jpg')
        grayImg.thumbnail((20,20))
        img2 = np.asarray(grayImg)
        Image.fromarray(img2).show()
        return img2
#        print(grayImg.format)
#        print(grayImg.mode)
#        print(grayImg.size)
#        print(img.format)
#        print(img.mode)
#        print(img.size)
#        img.show()
#        data = image.imread('sample.jpg')
#        print(data.dtype)
#        print(data.shape)