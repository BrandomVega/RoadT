import numpy    as  np
import pandas   as  pd
import csv
import matplotlib.pyplot as plt
class Perceptron:
    def __init__(self, learningRate, umbral):
        self.weights = [0,0,0]
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
    
    def updateWeights(self, error,x):
        for i in range(2):
            self.weights[i] = self.weights[i] + self.learningRate*(error)*(x[i])

    def train(self,dataX,dataY,epochSize = 100):
        convergenceList = []
        for epoch in range(epochSize):
            print(f"=======EJECUTANDDO EPOCH {epoch+1}=========")
            sumError = 0
            for xOriginal,Y in zip(dataX, dataY):
                x = np.append(xOriginal, 1)
                y = Y
                Z = self.dotProduct(x)
                error = (y-Z)
                sumError+=pow(error,2)
                self.updateWeights(error, x)
            self.writeData(epoch, sumError)  
            sumError = sumError*0.5
            convergenceList.append(sumError)
            print(f"MSE epoch: {sumError}") 
            if sumError < self.umbral:
                break
        self.showConvergence(convergenceList)
        return self.weights, epoch
    
    def showConvergence(self, convergenceList):
        plt.plot(convergenceList)
        plt.show()

    def writeData(self, epoch, error):
        with open(self.trainFile,"a", encoding="utf8",newline='') as csvFile:
            writerFile = csv.writer(csvFile)
            writerFile.writerow([epoch, error, self.weights[0], self.weights[1],  self.weights[2]])

    def printProcess(self, X,Y,yCalculated, error):
        print(f"\n\n-Entrenando {X} = {Y} Ycalculado: {yCalculated} pesos: {self.weights}\nError: {error}")  



def loadData():
    dataset = pd.read_csv(r"C:\Users\Brandom\Desktop\datasets\multiple_linear_regression_dataset.csv")
    X = dataset.iloc[:, 0:2]
    Y = dataset.iloc[:, 2]
    dataPoints = pd.DataFrame({'dataPointX':X.iloc[:,0], 'dataPointY':X.iloc[:,1], 'dataPointZ':Y})

    dataPoints.to_csv("dataPoints.csv", header=True, index=False)
    return X.values,Y.values

def LR():
    umbral = 0.0001
    learningRate = 0.00001
    epoch = 1000

    X,Y = loadData()
    linearRegresion = Perceptron(learningRate=learningRate, umbral=umbral)
    pesos,iteraciones = linearRegresion.train(X,Y, epochSize=epoch) 


    import graph
LR()