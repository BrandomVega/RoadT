import numpy    as  np
import pandas   as  pd
import csv

import matplotlib.pyplot as plt
class Perceptron:
    def __init__(self, learningRate, umbral):
        self.weights = [0,0]
        self.learningRate = learningRate
        self.umbral = umbral  
        self.error = 0
        
        self.trainFile = r"./dataWeights.csv"
        self.clearFile()


    def clearFile(self):
        with open(self.trainFile, "w", newline='') as limpia:
            limpia.write("Epoch,Error,w1,w2\n")

    def dotProduct(self, X):
        return sum(valor * peso for valor, peso in zip(self.weights, X))
    
    def updateWeights(self, error,x):
        for i in range(2):
            self.weights[i] = self.weights[i] + self.learningRate*(error)*(x[i])

    def train(self,dataX,dataY,epochSize = 100):
        convergence = []
        for epoch in range(epochSize):
            print(f"=======EJECUTANDDO EPOCH {epoch+1}=========")
            sumError = 0
            for xOriginal,Y in zip(dataX, dataY):
                x = [xOriginal,1]
                y = Y
                Z = self.dotProduct(x)
                error = (y-Z)
                sumError+=pow(error,2)
                self.updateWeights(error, x)

            self.writeData(epoch, sumError)  
            sumError = sumError*0.5
            convergence.append(sumError)
            self.error = sumError

            print(f"MSE epoch: {sumError}") 
            if sumError < self.umbral:
                break
        self.showConvergence(convergence)
        return self.weights, epoch
    
    def writeData(self, epoch, error):
        with open(self.trainFile,"a", encoding="utf8",newline='') as csvFile:
            writerFile = csv.writer(csvFile)
            writerFile.writerow([epoch, error, self.weights[0], self.weights[1]])

    def printProcess(self, X,Y,yCalculated, error):
        print(f"\n\n-Entrenando {X} = {Y} Ycalculado: {yCalculated} pesos: {self.weights}\nError: {error}")  

    def showConvergence(self, convergence):
        plt.plot(convergence)
        plt.show()


def generateDataLinearRegresion(nPoints, pendiente, yIntercept, lenX = 1, error_std=0.01):
    X = lenX*np.random.rand(nPoints, 1)
    error = np.random.normal(0, error_std, size=(nPoints, 1))
    y = pendiente*X+yIntercept +error
    return X, y

def z_score_normalize(data):
    mean_value = sum(data) / len(data)
    std_dev = (sum((x - mean_value) ** 2 for x in data) / len(data)) ** 0.5

    normalized_data = [(x - mean_value) / std_dev for x in data]

    print(normalized_data)
    return normalized_data


def loadData(opt):
    if opt==1: 
        #Load
        dataset = pd.read_csv(r"C:\Users\Brandom\Desktop\datasets\Salary_Data.csv")
        X = dataset.iloc[:, 0]
        #X = z_score_normalize(X) #Normalize helps with convergence since data is "smaller"
        Y = dataset.iloc[:, 1]
        #Y = z_score_normalize(Y)
        dataPoints = pd.DataFrame({'dataPointX':X, 'dataPointY':Y})
    else:
        #Generate Dataset or load
        X,Y = generateDataLinearRegresion(nPoints=20, pendiente=0.6, yIntercept=2, error_std=0.1, lenX=15)
        X = X.flatten(); Y = Y.flatten()
        dataPoints = pd.DataFrame({'dataPointX':X, 'dataPointY':Y})

    dataPoints.to_csv("dataPoints.csv", header=True, index=False)
    return X,Y


def LR():
    umbral = 0.0001
    learningRate = 0.01
    epoch = 1000

    X,Y = loadData(1)
    print(X)
    linearRegresion = Perceptron(learningRate=learningRate, umbral=umbral)
    pesos,iteraciones = linearRegresion.train(X,Y, epochSize=epoch)
    
    import graph 



LR()