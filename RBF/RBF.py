import numpy as np
import matplotlib.pyplot as plt
import numpy.core.defchararray as defCharArr

x = np.arange(-1.0, 1.01, 0.01)

testData = np.genfromtxt("test.txt", dtype='str')
testData = defCharArr.replace(testData, ',', '.')
testData = testData.astype(float)
print("Входные значения:\n", testData)
print("\n")

plt.figure(1)
plt.plot(x, testData, 'b.')   

sigma = 0.5
eta = 0.1
centres = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
weights = [0.2, 0.5, 0.8, 0.1, 0.88, 0.5, 0.3, 0.22, -0.2]


def net(testVal, centres):
    return np.exp((-1 / (2 * sigma)) * ((centres - testVal) ** 2))


numberOfEvals = range(2000)
testDataLength = len(testData)
numberOfValues = np.arange(0, testDataLength, 1)

for j in numberOfEvals:
    for i in numberOfValues:
        netValue = net(x[i], centres)
        netOut = np.dot(netValue, weights)
        delta = testData[i] - netOut
        deltaWeights = eta * delta * netValue
        weights += deltaWeights

#print (weights)

X = np.arange(-1, 1.01, 0.01)
netCount = range(9)
xCount = range(testDataLength)

netFinalOut = []
netOutSum = 0.0
for j in xCount:
    for i in netCount:
        netValue = net(X[j], centres[i])
        netOut = np.dot(netValue, weights[i])
        netOutSum += netOut
    netFinalOut.append(netOutSum)
    netOutSum = 0.0

print("Выходные значения:", netFinalOut)

exportData = open("testOut.txt", "w")
exportData.write("\n".join(map(str, netFinalOut)))
exportData.close()

plt.figure(1)
plt.plot(X, netFinalOut, 'r-')
plt.grid(True)
plt.show()
