import numpy as np
import matplotlib.pyplot as plt
from PatternJitter import *
from Data import *

Ref = [10, 15, 22, 29, 34, 40, 45, 51]
Ref02 = [10, 14, 17]
# x_tilde_02 = [10, 14, 17]

# Finding: P(S_j | T_j)
def getSyncState(L, Reference, Target):
    ref = Reference
    givenT = getOmega(L, Target)
    # print('Reference Train: ', ref)

    index = 0
    upState = []
    upStateIndex = []

    syncStateMat = []
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

Target = x_tilde_02
syncStateMat = getSyncState(3, Ref02, Target)
print('Sync State Matrix: ')
print(syncStateMat, '\n')

def getInitSyncDist(InitDist, SyncState):

    P_S1 = []
    p1_sync = []
    p1_non_sync = []

    initDist = InitDist
    syncStateMat = SyncState
    nonSyncState = 0

    for j, prob in enumerate(initDist):
        syncState = syncStateMat[0][1][j]
        # print('Probability: ', j, prob)
        # print('Sync State: ', syncState)
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

def getNewP_S(P_S):
    PS_T = np.array(P_S).T
    new = []
    for row in PS_T:
        new.append(np.sum(row))
    new = np.array(new)
    return new

P_Smat = getInitSyncDist(initDist_02, syncStateMat)
print('Init P_S: ')
print(P_Smat, '\n')

# print('Transition Matrices: ')
# print(tDistMatrices_02)

def getP_S1(SyncState, SyncDist):
    P_S1 = []
    for i, row in enumerate(SyncDist):
        a = SyncState[0][i]
        b = np.array(row).T
        result = np.dot(a, b)
        P_S1.append(result)

    p_S1 = np.array(P_S1)
    return P_S1


P_S1 = getP_S1(syncStateMat, P_Smat)


def getZdist(which):
    a = []
    result = []
    for k, row in enumerate(syncStateMat[which-1]):
        a = np.dot(row, tDistMatrices_02[which-2])
        result.append(a)

    result = np.array(result)
    return result


def getSyncDist(which):

    S = which
    # P_S = []
    zDistMat = []
    temp01 = []
    temp02 = []
    P_Stest = []
    P_Smat = []
    P_Smat = getInitSyncDist(initDist_02, syncStateMat)

    for i in range(S-1):

        zDistMat = getZdist(i)

        print('Z Dist Mat:', i)
        print(zDistMat, '\n')

        print('P_Smat:', i)
        print(P_Smat, '\n')

        P_S = []

        for j, preZdist in enumerate(P_Smat):
            # print('Yo1: ', j, preZdist)
            for k, zDist in enumerate(zDistMat):
                # print('Yo2: ', k, zDist)
                result = np.dot(preZdist, zDist)
                matMult = np.multiply(preZdist, zDist)
                Sum = j+k

                if  Sum % (2+i) != 0:
                    # print('(k+j) mod S: ', k+j)
                    temp01.append(result)
                    temp02.append(matMult)
                    # print('Yo 1, result: ', result)

                    if len(temp01) == 2:

                        # print('Yo 1 Up, result: ', temp01)
                        # print(np.sum(temp01))
                        P_S.append(np.sum(temp01))
                        temp01 = []

                    if len(temp02) == 2:

                        P_Stest.append(np.sum(temp02, axis=0))
                        temp02 = []

                else:
                    # print('Yo 2, result: ', result)
                    P_S.append(result)
                    P_Stest.append(matMult)

            # print('\n')

        P_Stest = np.array(P_Stest)
        P_Smat = P_Stest
        P_Stest = []

        # print('New P_Smat: ')
        # print(P_Smat, '\n')

    return P_S


P_S = getSyncDist(3)
print('P(S1):', P_S1)
print('P_S:', P_S)
