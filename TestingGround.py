import numpy as np
import xlrd

netFirstLayer = np.dot(WeightsFirstLayer, TrainDataNumbers[i]) + Biase1
OutNetFirstLayer = Sigmoid(netFirstLayer)
netHiddenLayer = np.dot(WeightsHiddenLayer, OutNetFirstLayer) + Biase2
OutNetHiddenLayer = Sigmoid(netHiddenLayer)

print(OutNetHiddenLayer)
print(TrainDataClassificator[:, i])

delta = (OutNetHiddenLayer - TrainDataClassificator[:, i]) * dSigmoid(netHiddenLayer)
dWeightsHiddenLayer = np.dot(delta, OutNetFirstLayer.T)

delta = np.dot(WeightsHiddenLayer.T, delta) * dSigmoid(netFirstLayer)
dWeightsFirstLayer = np.dot(delta, TrainDataNumbers[i].T)

WeightsHiddenLayer -= learningRate * dWeightsHiddenLayer
WeightsFirstLayer -= learningRate * dWeightsFirstLayer






netFirstLayer = np.array([np.sum(WeightsFirstLayer[0] * TrainDataNumbers[i]), np.sum(WeightsFirstLayer[1] * TrainDataNumbers[i])])
        OutNetFirstLayer = Sigmoid(netFirstLayer)
        netHiddenLayer = np.array([np.sum(WeightsHiddenLayer[0] * OutNetFirstLayer[0]), np.sum(WeightsHiddenLayer[1] * OutNetFirstLayer[1])])
        OutNetHiddenLayer = Sigmoid(netHiddenLayer)

        delta = (OutNetHiddenLayer - TrainDataClassificator[:, i]) * dSigmoid(netHiddenLayer)
        dWeightsHiddenLayer = (delta * OutNetFirstLayer).reshape(2, 1)

        delta = np.array([np.sum(WeightsHiddenLayer[0] * delta), np.sum(WeightsHiddenLayer[1] * delta)]) * dSigmoid(netFirstLayer)
        dWeightsFirstLayer = (delta * TrainDataNumbers[i])

        print(WeightsFirstLayer)
        print(dWeightsHiddenLayer)

        WeightsHiddenLayer -= learningRate * dWeightsHiddenLayer
        WeightsFirstLayer -= learningRate * dWeightsFirstLayer