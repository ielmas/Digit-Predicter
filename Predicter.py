# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 08:00:24 2019

Description: After training the model and saving it, you are able to use it
for prediction of images.
    
@author: Ibrahim Elmas
"""

import pandas as pd
import numpy as np
from PIL import Image
from matplotlib import image




class Predicter:
    ## "h" : the value for input to sigmoid function
    def Sigmoid(self, h): ## sigmoid function for the calculating hypothesis
        return 1 / (1 + np.exp(-h))
    
    ## "path" : the saved theta values( MODEL)
    def loadModel(self, path): ## here you load the model.
        self.thetas = pd.read_csv(path, sep=',',header=None)
      
    ## input:
    ## "X" : array representation of the image. (it has to be 1 to 400 dimensions (1 row, 400 columns))
    ## output:
    ## "predicted" : predicted value
    def predict(self, X):     ## this is the actual prediction function computes probabilities.
        newX = np.zeros((1, 401))
        newX[0][0] = 1
        for i in range(400):
            if(i != 0):
                newX[0][i+1] = X[i]
        probs = newX.dot(np.transpose(self.thetas))
        probs = self.Sigmoid(probs)
        predicted = np.argmax(probs, 1)
        return predicted
        
    ## input:
    ## "path": the path to ımage to be predicted
    ## output:
    ## "img2": array representation of the image converted to grayscale.
    def loadImage(self, path):   ## here you upload the ımage to be predicted.
        img = Image.open(path)
        grayImg = img.convert(mode='L')
        grayImg.save('sampleGray.jpg')
        grayImg.thumbnail((20,20))
        img2 = np.asarray(grayImg)
        return img2
    
    