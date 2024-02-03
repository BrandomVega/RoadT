import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np

class perceptronPlot:
    def __init__(self, pathData, pathPerceptron, data, dimensions):
        self.pathPoints = pathData
        self.pathWeights= pathPerceptron
        self.data       = data
        self.dimensions = dimensions


        self.fig, self.ax = plt.subplots()
        #plt.style.use('fivethirtyeight')

    def readFile(self):
        #Extract Data points
        dataFile = pd.read_csv(self.pathPoints)
        dataPoints = dataFile.iloc[:, 0:self.dimensions]

        #Plot points
        if self.dimensions == 2:
            self.plotPoints2D(dataPoints)
        else:
            self.plotPoints3D(dataPoints)

        #Extract minimun and maximun axis for ploting adjust
        LimitsMinMax = []
        for i,axName in enumerate(dataPoints):
            axValues = dataPoints.iloc[:,i]
            minValue = min(axValues)
            maxValue = max(axValues)
            LimitsMinMax.append([minValue, maxValue])

        #Extract data for hyperplane
        weightsFile = pd.read_csv(self.pathWeights)
        weights = weightsFile.iloc[:, 2:2+self.dimensions]
        epoches = weightsFile.iloc[:, 0]
        error   = weightsFile.iloc[:, 1]

        self.plotPlane2D(epoches,weights,error,LimitsMinMax, dataPoints)

        


    def animatePlane(self, i, weights, listLimits, dataPoints, error):
        w1 = weights.iloc[i, 0]
        w2 = weights.iloc[i, 1]

        xLimits = listLimits[0]

        x_values = np.linspace(round(xLimits[0], 0), round(xLimits[1], 0), 100)
        y_values = w1*x_values+w2

        plt.cla()
        self.ax.set_title(f"Epoch {i} con MSE:{round(error[i],4)}")
        self.plotPoints2D(dataPoints)
        self.ax.plot(x_values, y_values, color='red', linewidth=2, label='Regresion')
        plt.legend()
    

    def plotPlane2D(self, epoches, weights,error,listLimits, dataPoints):
        totalIterations = list(epoches)[-1]
        animation = FuncAnimation(plt.gcf(), self.animatePlane, frames=range(totalIterations), fargs=(weights,listLimits, dataPoints,error), interval=100)
        plt.show()

    def plotPoints2D(self, dataPoints):
        x = dataPoints.iloc[:,0]
        y = dataPoints.iloc[:,1]

        self.ax.scatter(x,y,s=100, c='black',alpha=1)
        plt.xlabel("X1")
        plt.ylabel("X2")

    
    def plotPoints3D(self, dataPoints):
        x = dataPoints.iloc[:,0]
        y = dataPoints.iloc[:,1]
        z = dataPoints.iloc[:,2]
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(projection='3d')   
        self.ax.scatter(x,y,z, s=100, c='black')
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        self.ax.set_zlabel("Z")


def graphMain():
    print(f"Inicia programa")
    
    myPlot = perceptronPlot("dataPoints.csv", "dataWeights.csv", "Data", 2)
    myPlot.readFile()
graphMain()