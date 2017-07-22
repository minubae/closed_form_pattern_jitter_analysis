import numpy as np
import matplotlib.pyplot as plt
from progress import *
from generateX import *


def getSpikeTrainMat(L, R, N):
    spikeTrainMat = []
    for i in range(N):
        surrogate = getSpikeTrain(x_tilde, L, R, initDist, tDistMatrices)
        spikeTrainMat.append(surrogate)

    Tmat = np.array(spikeTrainMat)
    return Tmat

def getAmountSync(R, Tmat):
    s = 0
    S = []
    for j, Tj in enumerate(Tmat):
        # Check how many elements are equal in two arrays (R, T)
        # print('Tj: ', Tj)
        s = np.sum(x_tilde == np.array(Tj))
        S.append(s)
        # print('# Sync: ', s)
    return S

Tmat = getSpikeTrainMat(L, R, 1000)
print(Tmat)
S = getAmountSync(R, Tmat)
print('Amount_Synchrony: ', S)
plt.hist(S, bins='auto')
plt.show()
