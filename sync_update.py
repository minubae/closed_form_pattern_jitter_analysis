import numpy as np
from PatternJitter import *
from Data import *

Ref = [10, 15, 22, 29, 34, 40, 45, 51]

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

Target = x_tilde
syncStateMat = getSyncState(5, Ref, Target)
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

P_S1 = getInitSyncDist(initDist, syncStateMat)
print('P(S1): ')
print(P_S1, '\n')


givenT = getOmega(5, Target)
N = len(syncStateMat)
M = len(tDistMatrices)
where = 1

P_S0 = []
P_S1 = []
resutl = 0
P_S = getInitSyncDist(initDist, syncStateMat)

for k in range(where):

    print(syncStateMat[k+1])
    # print(tDistMatrices[k])
    n = len(tDistMatrices[k])

    P_S0 = []
    P_S1 = []
    print('Check P_S: ')
    print(P_S)

    for j in range(n):
        print(tDistMatrices[k][j])

        for i, prob_i in enumerate(tDistMatrices[k][j]):

            if syncStateMat[k+1][i] == 1:
                # print('Sync: ', prob_i)
                # print(syncStateMat[k+1][i])
                result = prob_i*P_S[1][i]
                # print('Result: ', result)

                P_S0.append(0)
                P_S1.append(result)


            else:
                # print('Non Sync: ', prob_i)
                # print(syncStateMat[k+1][i])
                result = prob_i*P_S[0][i]
                # print('Result: ', result)

                P_S0.append(result)
                P_S1.append(0)

        print('New P_S0', P_S0)
        print('New P_S1',P_S1)
        print('\n')

        break

    P_S = []
    P_S.append(P_S0)
    P_S.append(P_S1)
    P_S = np.array(P_S)




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

    zDist = []


    if AmoutSync != len_zDistBasis:

        add = []
        new = []
        get = []
        print('Hello There.')
        get = zDistBasis[S-(S-1)]
        # print(zDistBasis)

        get0 = []
        get1 = []
        get0 = zDistBasis[0]
        get1 = zDistBasis[1]
        newBasis0 = []
        newBasis1 = []

        for i in range(S-len_zDistBasis):
            # print(i)
            newBasis0 = np.concatenate((get0, ZeroPadding), axis=1)
            newBasis1 = np.concatenate((get1, ZeroPadding), axis=1)

            new = np.concatenate((ZeroPadding, get), axis=1)

            get = new
            get0 = newBasis0
            get1 = newBasis1
            # add.append(new)
            # add.append(np.concatenate((zDistBasis[i], ZeroPadding), axis=1))

        print(newBasis0)
        # print('add:')
        # add = np.array(add)
        # print(add)
        # add.append(np.array(new))
        # zDistBasis = np.array(add)
        # print('New Z Dist Basis: ')
        # print(zDistBasis)


            # for i in range(S-1):
            #
            #     add.append(np.concatenate((zDistBasis[i], ZeroPadding), axis=1))
            #
            # add.append(new)
            #
            # zDist = np.array(add)
            # zDistBasis = zDist

    else:

        zDist = zDistBasis

    return False
# print(getZdistMatrix(4, 5, tDistMatrices))
