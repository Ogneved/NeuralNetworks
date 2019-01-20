import numpy as np
import xlrd

TrainFile = xlrd.open_workbook("data_train_2_bits.xlsx")
TrainFileSheet = TrainFile.sheet_by_name("data_train_2_bits")
TrainDataNumbers = np.zeros(shape=(35, 1000), dtype=float)
for i in range(35):
    TrainDataNumbers[i] = TrainFileSheet.col_values(i)
TrainFile = xlrd.open_workbook("data_train_classificator.xlsx")
TrainFileSheet = TrainFile.sheet_by_name("data_train_classificator")
TrainDataClassificator = np.zeros(shape=(10, 1000), dtype=float)
for i in range(10):
    TrainDataClassificator[i] = TrainFileSheet.col_values(i)

learningRate = 0.9
Epochs = 1000


def Sigmoid(net):
    return 1/(1 + np.exp(-0.1*net))


def dSigmoid(net):
    return (0.1 * np.exp(-net * 0.1)) / ((np.exp(-net * 0.1) + 1) ** 2)


WeightsFirstLayer = np.random.uniform(size=(6, 35))
WeightsHiddenLayer = np.random.uniform(size=(10, 6))

for j in range(Epochs):
    netFirstLayer = np.dot(WeightsFirstLayer, TrainDataNumbers)
    OutNetFirstLayer = Sigmoid(netFirstLayer)
    netHiddenLayer = np.dot(WeightsHiddenLayer, OutNetFirstLayer)
    OutNetHiddenLayer = Sigmoid(netHiddenLayer)

    delta = (OutNetHiddenLayer - TrainDataClassificator) * dSigmoid(netHiddenLayer)
    dBiase2 = np.sum(delta)
    dWeightsHiddenLayer = np.dot(delta, OutNetFirstLayer.T)

    delta = np.dot(WeightsHiddenLayer.T, delta) * dSigmoid(netFirstLayer)
    dBiase1 = np.sum(delta)
    dWeightsFirstLayer = np.dot(delta, TrainDataNumbers.T)

    WeightsHiddenLayer -= learningRate * dWeightsHiddenLayer
    WeightsFirstLayer -= learningRate * dWeightsFirstLayer


def TestNetwork(FileName, SheetName, NumberOfInputNumbers):
    TestFile = xlrd.open_workbook(FileName)
    TestFileSheet = TestFile.sheet_by_name(SheetName)
    TestDataNumber = np.zeros(shape=(NumberOfInputNumbers, 35), dtype=int)
    OutNumbers = np.zeros(shape=(1, NumberOfInputNumbers), dtype=int)
    for i in range(35):
        TestDataNumber[:, i] = TestFileSheet.col_values(i)
    for j in range(NumberOfInputNumbers):
        netTest1 = Sigmoid(np.dot(WeightsFirstLayer, TestDataNumber[j]))
        netTest2 = Sigmoid(np.dot(WeightsHiddenLayer, netTest1))
        OutNumbers[0, j] = np.argmax(netTest2)
    print('Numbers after algorithm:', OutNumbers)


TestNetwork("data_a_test_3_bits.xlsx", "data_three_test_3_bytes", 10)
TestNetwork("data_b_test_3_bits.xlsx", "data_seven_test_3_bytes", 10)
TestNetwork("data_c_test_3_bits.xlsx", "data_four_test_3_bytes", 10)
TestNetwork("data_d_test_3_bits.xlsx", "data_six_test_3_bytes", 10)
TestNetwork("data_k_test_3_bits.xlsx", "data_one_test_3_bytes", 10)
TestNetwork("data_l_test_3_bits.xlsx", "data_five_test_3_bytes", 10)
TestNetwork("data_m_test_3_bits.xlsx", "data_zero_test_3_bytes", 10)
TestNetwork("data_o_test_3_bits.xlsx", "data_two_test_3_bytes", 10)
TestNetwork("data_s_test_3_bits.xlsx", "data_eight_test_3_bytes", 10)
TestNetwork("data_t_test_3_bit.xlsx", "data_nine_test_3_bytes", 10)