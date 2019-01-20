import numpy as np
import random


def RandomNumberGenerator():
    RandomNumbers = np.array(range(0, 63))
    random.shuffle(RandomNumbers)
    return RandomNumbers


def NoiseForNumber(DataTrainNumber):
    NoisedNumber = DataTrainNumber
    ErrorBits = np.zeros(shape=(NumberOfErrorBits, 1), dtype='int')
    for i in range(NumberOfErrorBits):
        ErrorBits[i] = RandomNumberGenerator()[i]
    for i in ErrorBits:
        NoisedNumber[i] = NoisedNumber[i] * (-1)
    return NoisedNumber


def RestoreNumber(NoisedNumber):
    StabilityCheck = False
    tempStabilityCheck = np.zeros(shape=(10, 1))
    CheckNumber = 1
    while StabilityCheck != True:
        NumberFirstlyCheck = 10
        for j in range(NumberFirstlyCheck):
            RandomNumbers = RandomNumberGenerator()
            for i in RandomNumbers:
                if np.dot(NoisedNumber, MemoryWeights[:, i].reshape(63, 1)) > 0 > NoisedNumber[i]:
                    NoisedNumber[i] = 1
                if np.dot(NoisedNumber, MemoryWeights[:, i].reshape(63, 1)) < 0 < NoisedNumber[i]:
                    NoisedNumber[i] = -1
            Check[CheckNumber] = NoisedNumber
            CheckNumber += 1
        for n in range(NumberFirstlyCheck):
            if np.all(Check[CheckNumber - 1] == Check[CheckNumber - 1 - n]):
                tempStabilityCheck[n] = 1
        if np.sum(tempStabilityCheck) == NumberFirstlyCheck:
            StabilityCheck = True
        RestoredNumber = NoisedNumber
    print('Number of iterations during stability check:', CheckNumber)
    return RestoredNumber


def PrintNumber(Number):
    tempNumber = np.zeros(63, dtype='str')
    for i in range(63):
        if Number[i] == (-1):
            tempNumber[i] = ' '
        else:
            tempNumber[i] = 'O'
    return print(tempNumber.reshape(7, 9).T)


def TestNetwork(DataTrainNumber, Number):
    print('\nNumber', Number, 'with', NumberOfErrorBits, 'error bits:')
    PrintNumber(NoiseForNumber(DataTrainNumber))
    print('\nNumber after algorithm:')
    PrintNumber(RestoreNumber(NoiseForNumber(DataTrainNumber)))


def LoadTestNumber(NameOfFileWithErrorNumber):
    DataNumberWithErrors = np.genfromtxt(NameOfFileWithErrorNumber, dtype='int')
    print('\nNumber with error bits:')
    PrintNumber(DataNumberWithErrors)
    print('\nNumber after algorithm:')
    PrintNumber(RestoreNumber(DataNumberWithErrors))



DataTrainNumber0 = np.genfromtxt("DataTrainNumber0.txt", dtype='int')
DataTrainNumber1 = np.genfromtxt("DataTrainNumber1.txt", dtype='int')
DataTrainNumber2 = np.genfromtxt("DataTrainNumber2.txt", dtype='int')

NumberOfErrorBits = 10


Check = np.empty(shape=(200000, 63), dtype='int')
MemoryWeights = DataTrainNumber0 * DataTrainNumber0.reshape(63, 1) + DataTrainNumber2 * DataTrainNumber2.reshape(63, 1)
np.fill_diagonal(MemoryWeights, 0)

#TestNetwork(DataTrainNumber0, '0')
#TestNetwork(DataTrainNumber1, '1')
#TestNetwork(DataTrainNumber2, '2')

LoadTestNumber("V1.txt")