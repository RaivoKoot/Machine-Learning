# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:31:07 2018

@author: Raivo Koot
"""
from dataStorage import DataStorage

class SaveData:
    
    fileObject = None
    
    def __init__(self, fileName, mode):
        self.fileObject = open(fileName, mode)
        
    def writeThetaToFile(self, dataStorage):
        theta = dataStorage.thetaVector
        
        for parameter in theta:
            value = str(parameter[0])
            self.fileObject.write(value)
            self.fileObject.write("\n")
            
    def closeFile(self):
        self.fileObject.close()
        
    def readTheta(self, dataStorage):
        theta = dataStorage.thetaVector
        i = 0
        for line in self.fileObject:
            value = float(line)
            theta[i,0] = value
            i = i + 1
            
        dataStorage.thetaVector = theta
            
    
    
data = DataStorage(4)
fileReader = SaveData("theta.txt","r")

fileReader.readTheta(data)

print(data.thetaVector)