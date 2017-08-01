import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
from PatternJitter import *
from Data import *

Ref = [10, 15, 22, 29, 34, 40, 45, 51]
Ref_02 = [10, 14, 17]
Ref_03 = [10, 14, 17, 20]

# Finding: P(S_j | T_j)
def getSyncState(L, Reference, Target):

    index = 0
    length = 0

    ref = []
    tar = []
    givenT = []
    upState = []
    upStateIndex = []
    syncStateMat = []

    length = L
    tar = Target
    ref = Reference
    givenT = getOmega(length, tar)

    for j, Tj in enumerate(givenT):
        # print('All possible T[',j,']: ', Tj)
        # print('Reference[',j,']: ', ref[j])
        index = np.where(Tj == ref[j])[0]
        # print('Index: ', index)
        # print('Index Size: ', index.size, '\n')
        if index.size != 0:
            # np.asscalar(a): Convert an array of size 1 to its scalar equivalent
            index = np.asscalar(index)
            # upStateIndex.append(index)
            upState.append(Tj[index])
        else:
            # upStateIndex.append('No_Sync')
            upState.append(0)

        n = len(Tj)

        stateMat = []
        syncState = []
        nonSyncState = []
        checkSync = upState[j]

        # print('checkSync:', checkSync)
        # print('\n')
        for i in range(n):
            if Tj[i] == checkSync:
                # print('Check[', j,']: ', checkSync)
                # print('Tj[',i,']: ', Tj[i], 'is SYNCRONY.')
                syncState.append(1)
                nonSyncState.append(0)
            else:
                # print('Check[', j,']: ', checkSync)
                # print('Tj:', Tj[i], 'is not Synch.')
                syncState.append(0)
                nonSyncState.append(1)

        # print('yo 00:', nonSyncState)
        # print('yo 01:', syncState)

        stateMat.append(nonSyncState)
        stateMat.append(syncState)
        # print('Wow: ', stateMat)
        syncStateMat.append(np.array(stateMat))
    # print(upState)
    syncStateMat = np.array(syncStateMat)
    return syncStateMat


def getInitSyncDist(InitDist, SyncState):

    P_S1 = []
    initD = []
    p1_sync = []
    p1_non_sync = []
    syncStateMat = []

    initD = InitDist
    syncStateMat = SyncState
    nonSyncState = 0

    for j, prob in enumerate(initD):

        syncState = syncStateMat[0][1][j]

        if syncState == 0:

            p1_non_sync.append(prob*1)
            p1_sync.append(prob*nonSyncState)

        else:

            p1_non_sync.append(prob*0)
            p1_sync.append(prob*syncState)

    P_S1.append(p1_non_sync)
    P_S1.append(p1_sync)

    P_S1 = np.array(P_S1)
    return P_S1


def getP_S1(SyncState, SyncDist):

    a = 0
    result = 0

    b = []
    P_S1 = []
    syncS = []
    syncD = []

    syncD = SyncDist
    syncS = SyncState

    for i, row in enumerate(syncD):
        a = syncS[0][i]
        b = np.array(row).T
        result = np.dot(a, b)
        P_S1.append(result)

    p_S1 = np.array(P_S1)
    return P_S1


def getZdist(tDistMatrix, which):

    a = []
    index = 0
    result = []
    tDistMat = []

    index = which
    tDistMat = tDistMatrix
    # print('which', which)
    for k, row in enumerate(syncStateMat[index+1]):
        # print('yo: ', row)
        a = np.dot(tDistMat[index], np.array(row).T)
        result.append(a)

    result = np.array(result)
    return result


def getSyncDist(which, P_Smat, tDistMatrices):

    S = 0
    temp01 = []
    temp02 = []
    tDistM = []
    P_Stemp = []
    zDistMat = []

    S = which
    P_Sm = P_Smat
    tDistM = tDistMatrices

    for i in range(S-1):

        zDistMat = getZdist(tDistM, i)

        print('Z Dist Mat:', i)
        print(zDistMat, '\n')

        print('P_Smat:', i)
        print(P_Sm, '\n')

        P_S = []

        for j, preZdist in enumerate(P_Sm):
            # print('Yo1: ', j, preZdist)
            for k, zDist in enumerate(zDistMat):
                # print('Yo2: ', k, zDist)
                result = np.dot(zDist, np.array(preZdist).T)
                matMult = np.multiply(preZdist, zDist)
                Sum = j+k

                if  Sum % (2+i) != 0:
                    # print('(k+j) mod S: ', k+j)
                    temp01.append(result)
                    temp02.append(matMult)
                    # print('matMult: ', matMult)
                    # print('Yo 1, result: ', result)

                    if len(temp01) == 2:
                        # print('Yo 1 Up, result: ', temp01)
                        # print(np.sum(temp01))
                        P_S.append(np.sum(temp01))
                        temp01 = []

                    if len(temp02) == 2:

                        P_Stemp.append(np.sum(temp02, axis=0))
                        temp02 = []

                else:
                    # print('Yo 2, result: ', result)
                    P_S.append(result)
                    P_Stemp.append(matMult)
                    # print('matMult: ', matMult)

        P_Sm = np.array(P_Stemp)
        P_Stemp = []

    return P_S

Reference = Ref_03
Target = x_tilde_03
syncStateMat = getSyncState(3, Reference, Target)
print('Sync State Matrix: ')
print(syncStateMat, '\n')

P_Smat = getInitSyncDist(initDist_03, syncStateMat)
P_S1 = getP_S1(syncStateMat, P_Smat)
print('Init P_S: ')
print(P_Smat, '\n')

S_TrainN = len(Target)
P_S = getSyncDist(S_TrainN, P_Smat, tDistMatrices_03)
print('P(S1):', P_S1)
print('P(S',S_TrainN,'): ', P_S)

plt.plot(P_S, 'ro')
plt.axis([0, S_TrainN, 0, 1])
plt.show()
