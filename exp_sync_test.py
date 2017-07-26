import numpy as np
import matplotlib.pyplot as plt
from progress import *
from generateX import *
from Data import *

# x_tilde = [10, 15, 22, 29, 34, 40, 45, 51]
# L = 5
# R = 4

def getSpikeTrainMat(L, R, obsX, N):
    spikeTrainMat = []
    for i in range(N):
        print('[[[[[[[Spike Train Index: ', i,']]]]]]]')
        surrogate = getSpikeTrain(obsX, L, R, initDist, tDistMatrices)
        spikeTrainMat.append(surrogate)

    Tmat = np.array(spikeTrainMat)
    return Tmat

def getAmountSync(obsX, Tmat):
    s = 0
    S = []
    for j, Tj in enumerate(Tmat):
        # Check how many elements are equal in two arrays (R, T)
        # print('Tj: ', Tj)
        s = np.sum(obsX == np.array(Tj))
        S.append(s)
        # print('# Sync: ', s)
    return S

Tmat = getSpikeTrainMat(5, 4, x_tilde, 1000)
print(Tmat)
S = getAmountSync(x_tilde, Tmat)
print('Amount_Synchrony: ', S)
plt.hist(S, bins='auto')
plt.show()
