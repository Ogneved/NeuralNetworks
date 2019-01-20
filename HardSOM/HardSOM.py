import numpy as np
import matplotlib.pyplot as plt

learningRate = 0.01
Epochs = 50

Dimentions = 2
NumberOfClasters = 100
testDataXY = np.genfromtxt("4_KZ_setka.txt", dtype='float')
testDataXY = testDataXY.T
testDataX = testDataXY[0]
testDataY = testDataXY[1]
#NumberOfTrainVectors = np.size(testDataX)
NumberOfTrainVectors = 10000

#inputTrainVectors = np.array([testDataX, testDataY])
inputTrainVectors = np.random.uniform(-1, 1, size=(Dimentions, NumberOfTrainVectors))
WeightsOfClasters = np.random.uniform(-1, 1, size=(Dimentions, NumberOfClasters))

DistanceVectorToClaster = np.zeros(shape=(Dimentions, 1))
SqrDistanceVectorToClaster = np.zeros(shape=(NumberOfClasters, 1))


def SqrDistanceVectorToClasters(CurrentTrainVector):
    for CurrentClaster in range(NumberOfClasters):
        for i in range(Dimentions):
            DistanceVectorToClaster[i] = (WeightsOfClasters[i][CurrentClaster] - inputTrainVectors[i][CurrentTrainVector]) ** 2
        SqrDistanceVectorToClaster[CurrentClaster] = np.sum(DistanceVectorToClaster)
    return SqrDistanceVectorToClaster


def DeltaWeightsOfClasters(CurrentTrainVector):
    SqrDistanceVectorToClasters(CurrentTrainVector)
    DeltaWeightsOfClaster = np.zeros(shape=(Dimentions, NumberOfClasters))
    for CurrentClaster in range(NumberOfClasters):
        if SqrDistanceVectorToClaster[CurrentClaster] == min(SqrDistanceVectorToClaster):
            for i in range(Dimentions):
                DeltaDistance = inputTrainVectors[i][CurrentTrainVector] - WeightsOfClasters[i][CurrentClaster]
                DeltaWeightsOfClaster[i, CurrentClaster] = DeltaDistance
    return DeltaWeightsOfClaster


def PlotGraph(NumberOfFigure, NumberOfPoints, InputValues, Color, Marker):
    plt.figure(NumberOfFigure)
    for i in range(NumberOfPoints):
        plt.plot(InputValues[0][i], InputValues[1][i], color=Color, marker=Marker)
    plt.grid(True)


PlotGraph(1, NumberOfTrainVectors, inputTrainVectors, 'green', '.')
PlotGraph(1, NumberOfClasters, WeightsOfClasters, 'black', 'o')
plt.title('Before')

for i in range(Epochs):
    for CurrentTrainVector in range(NumberOfTrainVectors):
        WeightsOfClasters = WeightsOfClasters + learningRate * DeltaWeightsOfClasters(CurrentTrainVector)
    if i == Epochs/2:
        PlotGraph(2, NumberOfClasters, WeightsOfClasters, 'darkgray', 'd')
    print(i)
print(WeightsOfClasters)

PlotGraph(2, NumberOfTrainVectors, inputTrainVectors, 'green', '.')
PlotGraph(2, NumberOfClasters, WeightsOfClasters, 'red', 'o')
plt.title('After')
plt.show()