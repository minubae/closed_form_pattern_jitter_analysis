import numpy as np
from PatternJitter import *
from Data import *

Ref = [10, 15, 22, 28, 34, 42, 44, 49]

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
        syncState = []
        checkSync = upState[j]
        # print('\n')
        for i in range(n):
            if Tj[i] == checkSync:
                # print('Check[', j,']: ', checkSync)
                # print('Tj[',i,']: ', Tj[i], 'is SYNCRONY.')
                syncState.append(1)
            else:
                # print('Check[', j,']: ', checkSync)
                # print('Tj:', Tj[i], 'is not Synch.')
                syncState.append(0)

        syncStateMat.append(syncState)
    # print(upState)
    syncStateMat = np.array(syncStateMat)
    return syncStateMat

syncStateMat = getSyncState(5, Ref, x_tilde)
print('Sync State Matrix: ')
print(syncStateMat, '\n')
# print('Init Dist (p(t1)):')
# print(initDist)
# print('Transition Matrix: ')
# print(tDistMatrices, '\n')

def getInitSyncDist(InitDist, SyncState):

    P_S1 = []
    p1_sync = []
    p1_non_sync = []

    initDist = InitDist
    syncStateMat = SyncState
    nonSyncState = 0

    for j, prob in enumerate(initDist):
        syncState = syncStateMat[0][j]
        # print('Probability: ', j, prob)
        # print('Sync State: ', syncState)
        if syncState == 1:
            p1_non_sync.append(prob*0)
            p1_sync.append(prob*syncState)
        else:
            p1_non_sync.append(prob*1)
            p1_sync.append(prob*nonSyncState)

    P_S1.append(p1_non_sync)
    P_S1.append(p1_sync)

    P_S1 = np.array(P_S1)
    return P_S1

p_s1 = getInitSyncDist(initDist, syncStateMat)
print('P(S1): ')
print(p_s1, '\n')
# print(syncStateMat, '\n')

def getZeroPadding(Length):

    L = 0
    Zero = []
    L = Length
    # Create numpy.zeros(shape, dtype, order)
    zero_k = np.zeros(L, dtype=np.int)

    for i in range(L):
        Zero.append(zero_k)

    Zero = np.array(Zero)
    return Zero

# print(getZeroPadding(5))


# P(Z_2 | Z_1) <= Basis for Trnasition Matrices for Z = (Amount_S, Tj)
def getZdistBasis(tMatrix, where, Length):

    Sync = []
    NonSync = []
    ZeroPadding = []

    zDistBasis = []
    zDistBasis0 = []
    zDistBasis1 = []

    which = where
    m = which+1
    L = Length
    for i, Pi in enumerate(tMatrix[which]):

        nonSync_k = []
        sync_k = []

        for k, prob in enumerate(Pi):

            if syncStateMat[m][k] == 0:

                nonSync_k.append(prob)
                sync_k.append(0)

            else:

                nonSync_k.append(0)
                sync_k.append(prob)

        NonSync.append(nonSync_k)
        Sync.append(sync_k)

    ZeroPadding = getZeroPadding(L)

    Basis = np.concatenate((np.array(NonSync), np.array(Sync)), axis=1)
    zDistBasis0 = np.concatenate((Basis, ZeroPadding), axis=1)
    zDistBasis1 = np.concatenate((ZeroPadding, Basis), axis=1)

    zDistBasis.append(zDistBasis0)
    zDistBasis.append(zDistBasis1)
    zDistBasis = np.array(zDistBasis)

    return zDistBasis

# print(getZdistBasis(tDistMatrices, 5))

def getZdistMatrix(S, L, tDistMat):

    AmoutSync = S
    zDistBasis = getZdistBasis(tDistMat, S-2, L)
    ZeroPadding = getZeroPadding(L)
    len_zDistBasis = len(zDistBasis)

    new = []
    add = []
    zDist = []

    if AmoutSync != len_zDistBasis:

        new = np.concatenate((ZeroPadding, zDistBasis[S-2]), axis=1)

        for i in range(S-1):

            add.append(np.concatenate((zDistBasis[i], ZeroPadding), axis=1))

        add.append(new)

        zDist = np.array(add)
        return zDist

    else:

        zDist = zDistBasis
        return zDist

print('New:')
print(getZdistMatrix(3, 5, tDistMatrices))
