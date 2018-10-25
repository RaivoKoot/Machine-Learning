# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 13:26:10 2018

@author: Raivo Koot
"""

from regressionbasics import RegressionBasics
from dataStorage import DataStorage
from saveData import SaveData

# initialize objects
regressionObject = RegressionBasics()
dataStorage = DataStorage()

# import data from pc and set up different matrices
regressionObject.loadData("mlr03.xls",dataStorage)

#compute the cost of our hypothesis
#cost = regressionObject.computeCost(dataStorage)

#print(cost)

regressionObject.gradientDescent(dataStorage, 0.000003)

print("error")
print(regressionObject.computeCost(dataStorage))    
print(dataStorage.thetaVector)

dataSave = SaveData("theta.txt","w")

dataSave.writeThetaToFile(dataStorage)
dataSave.closeFile()