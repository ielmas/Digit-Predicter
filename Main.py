# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 07:58:00 2019
Description: Main class that uses both Trainer and Predicter class for 
debugging.
@author: OrduLou
"""

from Trainer import Trainer

class main:
    
    path = 'sampleDatabase.mat'
    trainer = Trainer(path)
    trainer.loadDataset()
    trainer.printDataset()
    trainer.costFunction()