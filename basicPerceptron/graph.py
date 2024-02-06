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


        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111,projection='3d')
        #plt.style.use('fivethirtyeight')

    def readFile(self):
        #Extract Data points
        dataFile = pd.read_csv(self.pathPoints)
        dataPoints = dataFile.iloc[:, 0:self.dimensions]

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

        self.plotPlane3D(epoches,weights,error,LimitsMinMax, dataPoints)

    


    def animatePlane(self, i, weights, listLimits, dataPoints, error):
        w1 = weights.iloc[i, 0]
        w2 = weights.iloc[i, 1]
        w3 = weights.iloc[i, 2]


        xLimits = listLimits[0]
        yLimits = listLimits[1]

        x_values = np.linspace(round(xLimits[0]-1, 0), round(xLimits[1]+1, 0), 100)
        
        
        y_values = np.linspace(round(yLimits[0]-1, 0), round(yLimits[1]+1, 0), 100)
        X, Y = np.meshgrid(x_values, y_values)
        
        z_values = np.array(w1*X + w2*Y + w3)
        zFunction = np.where(z_values > 0, 1, 0)
        #print(f"Epoch {i}")
        #print(f"Valor para 0,0 = {w1*0 + w2*0 + w3} = 0")
        #print(f"Valor para 1,0 = {w1*1 + w2*0 + w3} = 0")
        #print(f"Valor para 0,1 = {w1*0 + w2*1+ w3} = 0")
        #print(f"Valor para 0,0 = {w1*1 + w2*1 + w3} = 1")

        plt.cla()
        self.ax.set_title(f"Epoch {i} con MSE:{round(error[i],4)}\n[{w1},{w2},{w3}]")
        self.plotPoints3D(dataPoints)
        self.ax.plot_surface(X, Y, zFunction, alpha=0.5)
        #self.ax.plot(x_values,y_values)
    

    def plotPlane3D(self, epoches, weights,error,listLimits, dataPoints):
        totalIterations = len(epoches)
        print(totalIterations)
        animation = FuncAnimation(plt.gcf(), self.animatePlane, frames=range(totalIterations), fargs=(weights,listLimits, dataPoints,error), interval=100)
        plt.get_current_fig_manager().full_screen_toggle()
        plt.show()


    
    def plotPoints3D(self, dataPoints):
        x = dataPoints.iloc[:,0]
        y = dataPoints.iloc[:,1]
        z = dataPoints.iloc[:,2]
        colors = ['blue' if val == 1 else 'green' for val in z]
        self.ax.scatter(x,y,z, s=100, c=colors)
        self.ax.set_xlabel("X1")
        self.ax.set_ylabel("X2")
        self.ax.set_zlabel("Z")


def graphMain():
    print(f"Inicia programa")
    
    myPlot = perceptronPlot("dataPoints.csv", "dataWeights.csv", "Data", 3)
    myPlot.readFile()
graphMain()