import numpy as np
import matplotlib.pyplot as plt
from PatternJitter import *
from Surrogate import *
from Data import *

# x_tilde = [10, 15, 22, 29, 34, 40, 45, 51]
# L = 5
# R = 4
Ref = [8, 13, 19, 28, 34, 42, 44, 49]
Ref02 = [10, 14, 17]

def getSpikeTrainMat(L, R, obsX, N):
    spikeTrainMat = []
    for i in range(N):
        print('[[[[[[[Spike Train Index: ', i,']]]]]]]')
        surrogate = getSpikeTrain(obsX, L, R, initDist_02, tDistMatrices_02)
        spikeTrainMat.append(surrogate)

    Tmat = np.array(spikeTrainMat)
    return Tmat

def getAmountSync(Reference, Target):
    s = 0
    S = []
    Ref = []
    Tmat = []
    Ref = Reference
    Tmat = Target
    for j, Tj in enumerate(Tmat):
        # Check how many elements are equal in two arrays (R, T)
        # print('Tj: ', Tj)
        s = np.sum(Ref == np.array(Tj))
        S.append(s)
        # print('# Sync: ', s)
    return S

Tmat = getSpikeTrainMat(3, 2, x_tilde_02, 10000)
print('Spike Trains: ')
print(Tmat)
print('Reference Train: ')
print(Ref)

S = getAmountSync(Ref02, Tmat)
print('Amount_Synchrony: ', S)
plt.hist(S, bins='auto')
plt.show()
