# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 12:22:08 2018

@author: Raivo Koot
"""
import numpy as np

class DataStorage:

    def __init__(self, thetaRows = 1):
        self.thetaVector = np.zeros((thetaRows, 1))
        self.featureMatrix = None
        self.expectedOutputVector = None
