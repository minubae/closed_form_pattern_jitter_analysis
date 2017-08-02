import numpy as np
import matplotlib.pyplot as plt

from ClosedSynchrony import *
from PatternJitter import *
from TransitMatrix import *
from Surrogate import *
from Data import *

# x_tilde = [10, 15, 22, 29, 34, 40, 45, 51]
# L = 5
# R = 4
Ref = [8, 13, 19, 28, 34, 42, 44, 49]
Ref_02 = [10, 14, 17]
Ref_03 = [10, 14, 17, 20]

def getSpikeTrainMat(L, R, obsX, initDist, tDistMatrices, N):

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
        surrogate = getSurrogate(ObsTar, length, hisLen, initD, tDistMat)
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

L = 5
R = 2
fRate = 40
Size = 100
spikeData = getSpikeData(Size, fRate)
spikeTrain = getSpikeTrain(spikeData)

print('Spike Data: ')
print(spikeData)

N = len(spikeTrain)
initDist = getInitDist(L)
tDistMatrices = getTransitionMatrices(L, N)
ref = getReference(Size, N)

print('Initial Distribution: ')
print(initDist)
print('Transition Matrices: ')
print(tDistMatrices)


################################################################
# Compute the Synchrony Distribution by Monte Carlo resamples
################################################################
Tmat = getSpikeTrainMat(L, R, spikeTrain, initDist, tDistMatrices, 1000)
print('Spike Trains: ')
print(Tmat)


print('Reference: ')
print(ref)
print('Target: ')
print(spikeTrain)


S = getAmountSync(ref, Tmat)
print('Amount_Synchrony: ', S)
plt.hist(S, bins='auto')
plt.show()

################################################
# Compute the Closed Synchrony Distribution
################################################

syncStateMat = getSyncState(L, ref, spikeTrain)
print('Sync State Matrix: ')
print(syncStateMat, '\n')

P_Smat = getInitSyncDist(initDist, syncStateMat)
P_S1 = getP_S1(syncStateMat, P_Smat)
print('Init P_S: ')
print(P_Smat, '\n')


P_S = getSyncDist(N, P_Smat, syncStateMat, tDistMatrices)
print('P(S1):', P_S1)
print('P(S',N,'): ', P_S)
print('Area of Sync Dist (S',N,'): ', np.sum(P_S))

print('Reference: ')
print(ref)

print('Target: ')
print(spikeTrain)


plt.plot(P_S, 'ro')
# plt.axis([0, N, 0, 1])
plt.show()
