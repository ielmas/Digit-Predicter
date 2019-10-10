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
        
    def printDataset(self):
        print(self.dataset)
        