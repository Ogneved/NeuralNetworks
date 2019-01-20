import numpy as np

inputValuesX = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
targetValuesY = np.array([0, 1, 1, 0]).reshape(1, 4)
learningRate = 1
Epochs = range(5000)


def Sigmoid(net):
    return 1 / (1 + np.exp(-net))


def dSigmoid(net):
    return Sigmoid(net) * (1 - Sigmoid(net))


Weights12 = np.random.uniform(size=(2, 2))
Biase1 = np.random.uniform()
Weights3 = np.random.uniform(size=(1, 2))
Biase2 = np.random.uniform()

for i in Epochs:
    net12 = np.dot(Weights12, inputValuesX) + Biase1
    OutNet12 = Sigmoid(net12)
    net3 = np.dot(Weights3, OutNet12) + Biase2
    OutNet3 = Sigmoid(net3)

    delta = (OutNet3 - targetValuesY) * dSigmoid(net3)
    dBiase2 = np.sum(delta)
    dWeights3 = np.dot(delta, OutNet12.T)

    delta = np.dot(Weights3.T, delta) * dSigmoid(net12)
    dBiase1 = np.sum(delta)
    dWeights12 = np.dot(delta, inputValuesX.T)

    Weights3 -= learningRate * dWeights3
    Biase2 -= learningRate * dBiase2
    Weights12 -= learningRate * dWeights12
    Biase1 -= learningRate * dBiase1

testInputValuesX = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

netTest1 = Sigmoid(np.dot(Weights12, testInputValuesX) + Biase1)
netTest2 = Sigmoid(np.dot(Weights3, netTest1) + Biase2)
testOut = np.where(netTest2 > 0.5, 1, 0).squeeze()
print(netTest2.reshape(4, 1))
print(testOut)
