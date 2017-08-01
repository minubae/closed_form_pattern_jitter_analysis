import numpy as np
import matplotlib.pyplot as plt
from PatternJitter import *
from Surrogate import *
from Data import *

# x_tilde = [10, 15, 22, 29, 34, 40, 45, 51]
# L = 5
# R = 4
Ref = [8, 13, 19, 28, 34, 42, 44, 49]
Ref_02 = [10, 14, 17]
Ref_03 = [10, 14, 17, 20]

def getSpikeTrainMat(L, R, obsX,initDist, tDistMatrices, N):

    n = 0
    length = 0
    hisLen = 0

    n = N
    length = L
    hisLen = R

    initD = []
    ObsTar = []
    tDistMat = []
    spikeTrainMat = []

    initD = initDist
    tDistMat = tDistMatrices
    ObsTar = obsX

    for i in range(N):
        print('[[[[[[[Spike Train Index: ', i,']]]]]]]')
        surrogate = getSpikeTrain(ObsTar, length, hisLen, initD, tDistMat)
        spikeTrainMat.append(surrogate)

    Tmat = np.array(spikeTrainMat)
    return Tmat

def getAmountSync(Reference, Target):
    s = 0
    S = []
    Ref = []
    Tmat = []
    ref = Reference
    Tmat = Target
    for j, Tj in enumerate(Tmat):
        # Check how many elements are equal in two arrays (R, T)
        # print('Tj: ', Tj)
        s = np.sum(ref == np.array(Tj))
        S.append(s)
        # print('# Sync: ', s)
    return S

Tmat = getSpikeTrainMat(3, 2, x_tilde_03, initDist_03, tDistMatrices_03, 1000)
print('Spike Trains: ')
print(Tmat)
print('Reference Train: ')
print(Ref_03)

S_TrainN = len(x_tilde_03)
S = getAmountSync(Ref_03, Tmat)
print('Amount_Synchrony: ', S)
plt.hist(S, bins='auto')
# plt.axis([0, S_TrainN])
plt.show()
