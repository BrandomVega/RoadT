import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
import random
class Perceptron:
    def __init__(self, learningRate, umbral):
        self.weights = [-random.uniform(0,1),-random.uniform(0,1),-random.uniform(0,1)]
        #self.weights = [0,0,0]
        self.learningRate = learningRate
        self.umbral = umbral  
        self.error = 0
        
        self.trainFile = r"./dataWeights.csv"
        self.clearFile()

    def clearFile(self):
        with open(self.trainFile, "w", newline='') as limpia:
            limpia.write("Epoch,Error,w1,w2,w3\n")

    def dotProduct(self, X):
        return sum(valor * peso for valor, peso in zip(self.weights, X))
    
    def updateWeights(self, gradient, epoch,x):
        print(f"    Actualizando pesos")
        for i in range(len(self.weights)):
            print(f"    w[{i}] = {self.weights[i]} ==> {self.weights[i] + self.learningRate*gradient*(x[i])}")
            self.weights[i] = self.weights[i] + self.learningRate*gradient*(x[i])
        
    def reLu(self, X):
        if round(self.dotProduct(X)) > 0:
            return 1
        else:
            return 0


    def train(self,dataX,dataY,epochSize = 10000):
        self.writeData(0, 0)
        for epoch in range(epochSize):
            print(f"=======EJECUTANDDO EPOCH {epoch+1}=========")
            for xOriginal,Y in zip(dataX, dataY):
                x = np.append(xOriginal, 1)
                y = Y
                Z = self.dotProduct(x)
                yCalculated = Z if Z >= self.umbral else 0
                error = y-yCalculated
                
                if Z > 0:
                    gradiente = 2*(Y - yCalculated) #-2(Y-y')
                elif Z < 0:
                    gradiente = 0
                self.printProcess(x,y,yCalculated,error, gradiente)
                if error != 0:
                    self.updateWeights(error, epoch, x)
            self.writeData(epoch, error)
  
        return self.weights, epoch


    def writeData(self, epoch, error):
        with open(self.trainFile,"a", encoding="utf8",newline='') as csvFile:
            writerFile = csv.writer(csvFile)
            writerFile.writerow([epoch, error, self.weights[0], self.weights[1], self.weights[2]])

    def printProcess(self, X,Y,yCalculated, error, gradiente):
        print(f"\n-Entrenando {X} = {Y} Ycalculado: {yCalculated} pesos: {self.weights} Error: {error} Grad: {gradiente}")  

    def showConvergence(self, convergence):
        plt.plot(convergence)
        plt.show()

def loadData():
    dataset = pd.read_csv(r"C:\Users\Brandom\Documents\roadTransformer\basicPerceptron\dataset.csv")
    X = dataset.iloc[:, 0:2]
    Y = dataset.iloc[:, 2]
    dataPoints = pd.DataFrame({'dataPointX':X.iloc[:,0], 'dataPointY':X.iloc[:,1], 'dataPointZ':Y})

    dataPoints.to_csv("dataPoints.csv", header=True, index=False)
    return X.values,Y.values

def main():
    X,Y = loadData()
    percep = Perceptron(0.1, 0)
    
    pesos, epoch =percep.train(X,Y, epochSize=100)

    w1,w2,b = pesos[0],pesos[1],pesos[2]

    

    print(f"0,0,0 = {percep.reLu([0,0,1])} ")
    print(f"0,1,0 = {percep.reLu([0,1,1])}")
    print(f"1,0,0 = {percep.reLu([1,0,1])}")
    print(f"1,1,1 = {percep.reLu([1,1,1])}")

    import graph
main()