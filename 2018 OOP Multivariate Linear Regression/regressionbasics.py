# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 19:56:38 2018

@author: Raivo Koot
"""
from matrixoperations import MatrixOperations

class RegressionBasics:
    mo = None
    
    def __init__(self):
        self.mo = MatrixOperations()
    
    def getPredictedYVector(self, dataStorage):
        featureMatrix = dataStorage.featureMatrix
        thetaVector = dataStorage.thetaVector
        
        outputVector = featureMatrix @ thetaVector
        return outputVector
    
    def getDifferences(self, dataStorage):
        predictionVector = self.getPredictedYVector(dataStorage)
        expectedVector = dataStorage.expectedOutputVector
        
        differencesVector = self.mo.subtractElements(predictionVector,expectedVector)
        return differencesVector
    
    def getSquaredDifferences(self, dataStorage):
        differencesVector = self.getDifferences(dataStorage)
        
        squaredDifferencesVector = self.mo.squareAllElements(differencesVector)
        return squaredDifferencesVector
    
    def getAverageSquaredDifference(self, squaredDifferencesVector):
        sumOfError = self.mo.getSumOfElements(squaredDifferencesVector)
        
        numberTrainingExamples = self.mo.getAmountOfElements(squaredDifferencesVector)
        
        averageError = sumOfError / (2 * numberTrainingExamples)
        
        return averageError
        
    def computeCost(self, dataStorage):
        # import the matrices from data storage
        squaredDifferences = self.getSquaredDifferences(dataStorage)
        cost = self.getAverageSquaredDifference(squaredDifferences)
        
        return cost
    
    def loadData(self, fileName, dataStorage):
        # scan data from file into python
        dataFrame = self.mo.scanDataFrameFromFile(fileName)
        dataMatrix = self.mo.getMatrixFromDataFrame(dataFrame)
        
        # select only last column of data
        expectedOutput = self.mo.getYColumn(dataMatrix)
        
        # cut off y from data and add theta0
        featureMatrix = self.mo.cutOffYColumn(dataMatrix)
        featureMatrix = self.mo.addColumnOfOnes(featureMatrix)
        
        #generate initial theta vector
        theta = self.getThetaVector(featureMatrix)
        
        # save the matrices inside of dataStorage object
        dataStorage.expectedOutputVector = expectedOutput
        dataStorage.featureMatrix = featureMatrix
        dataStorage.thetaVector = theta
        
        
    def getThetaVector(self, featureMatrix):
        amountOfFeatures = featureMatrix.shape[1]
        return self.mo.getThetaVector(amountOfFeatures)
    
    def gradientDescent(self, dataStorage, learningRate):
        i = 0
        lastStepsCost = 999999999
        while True:
            i = i +1
            self.gradientDescentStep(dataStorage, learningRate)
            currentStepCost = self.computeCost(dataStorage)
            if ( (lastStepsCost - currentStepCost) < 0.001):
                break
            lastStepsCost = currentStepCost
            
        print(str(i) + " number of iterations")
        
    
    def gradientDescentStep(self, dataStorage, learningRate):
        featureMatrix = dataStorage.featureMatrix # features
        thetaVector = dataStorage.thetaVector # parameters
        
        differenceVector = self.getDifferences(dataStorage)
        
        parameterUpdates = []
        
        i = 0
        while i < thetaVector.shape[0]:
            #parameter = thetaVector[i][0]
            featuresOfParameter = self.mo.getColumnOfMatrix(featureMatrix, i)
            partialDerivatives = differenceVector * featuresOfParameter
            averageDerivative = self.mo.getSumOfElements(partialDerivatives)
            parameterUpdates.append(averageDerivative * learningRate)
            i = i+1
        
        parameterUpdates = self.mo.listToVector(parameterUpdates)
        #print("parameters")
        #print(parameterUpdates)
        #print()
        
        updatedTheta = thetaVector - parameterUpdates
        dataStorage.thetaVector = updatedTheta
        
        
  #  def partialDerivative(dataStorage):
  
  
  
  
  
  
  
  
        