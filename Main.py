# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 07:58:00 2019

Description: Main class that uses both Trainer and Predicter class for 
debugging.

@author: Ibrahim Elmas
"""
import numpy as np
from Trainer import Trainer
from Predicter import Predicter 
import scipy.io as sio
class main:
    
    path = 'sampleDatabase.mat' ## this is the dataset file that you want to train with.
    
    
    ##### TRAINING PART
    trainer = Trainer(path)     ## this path is parameter to the constructor
    trainer.loadDataset() 
    trainer.Train()
    trainer.Predict()
    
    ### getting dataset to predict
    dataset = sio.loadmat(path)
    X = dataset['X']
    y = dataset['y']
    
    #### the saved model is uploaded to predictor class.(Training o,automatically saves models to this file)
    pathOfModel = 'thetaValues.csv'
    
    #### the predictor class is able to use model now
    predicter = Predicter()
    predicter.loadModel(pathOfModel)
    
    #### any image to predict (of course it has to be digit :D)
    pathOfImage = 'sample.jpg'
    
    ### laod image function returns the array representation of the image,
    ### so that you can put in predic function
    matrix = predicter.loadImage(pathOfImage)
    
    ## you have to convert X to 1 to 400 dimensions.(predict fucntions is so arranged)
    X = np.zeros((400, 1))
    for i in range(20):
        for j in range(20):
            X[20*i+j][0] = matrix[j][i]
            
    prediction = predicter.predict(X)
    print(prediction)
    