import numpy as np
from PatternJitter import *
from Data import *

Ref = [9, 15, 22, 28, 34, 42, 44, 49]

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

def getInitSyncDist(InitDist, SyncState):

    P_S1 = []
    p1_sync = []
    p1_non_sync = []

    initDist = InitDist
    syncStateMat = SyncState

    for j, prob in enumerate(initDist):
        syncState = syncStateMat[0][j]
        # print('Probability: ', j, prob)
        # print('Sync State: ', syncState)
        if syncState == 1:
            p1_non_sync.append(prob*0)
            p1_sync.append(prob*syncState)
        else:
            p1_non_sync.append(prob)
            p1_sync.append(0)

    P_S1.append(p1_non_sync)
    P_S1.append(p1_sync)

    P_S1 = np.array(P_S1)
    return P_S1

p_s1 = getInitSyncDist(initDist, syncStateMat)

# print('Transition Matrix: ')
# print(tDistMatrices, '\n')

print('P(S1): ')
print(p_s1, '\n')

# print(syncStateMat, '\n')
# print(tDistMatrices[0])


NonSync = []
Sync = []
Zero = []

which = 1
m = which+1
for i, Pi in enumerate(tDistMatrices[which]):
    # print(syncStateMat[m])
    # print(Pi)

    nonSync_k = []
    sync_k = []
    zero_k = []
    for k, prob in enumerate(Pi):
        # print(syncStateMat[m][k])

        if syncStateMat[m][k] == 0:
            # print('No Sync')
            # print(prob)
            nonSync_k.append(prob)
            sync_k.append(0)
        else:
            # print('Sync')
            # print(prob)
            nonSync_k.append(0)
            sync_k.append(prob)
        zero_k.append(0)
    # print('nonSync_k: ', nonSync_k, '\n')
    NonSync.append(nonSync_k)
    Sync.append(sync_k)
    Zero.append(zero_k)

NonSync = np.array(NonSync)
Sync = np.array(Sync)
Zero = np.array(Zero)
print('Non Sync: ')
print(NonSync)

print('Sync: ')
print(Sync)

print('Zero: ')
print(Zero, '\n')

zDist = []
Basis = np.concatenate((NonSync, Sync), axis=1)
zDistBasis0 = np.concatenate((Basis, Zero), axis=1)
zDistBasis1 = np.concatenate((Zero, Basis), axis=1)

'''
zDist.append(zDist0)
zDist.append(zDist1)
zDist = np.array(zDist)
'''
# print('Transition Matrix for Zj: ')
# print(zDist)

# s = 2
zDist.append(zDistBasis0)
zDist.append(zDistBasis1)
zDist = np.array(zDist)

zDistNew = []
for j, row in enumerate(zDist):
    print(row, '\n')

    if j == 0:
        new = row
        new = np.concatenate((zDistBasis0, Zero), axis=1)
        print('New00: ')
        print(new)
        zDistNew.append(new)

    else:

        new = row
        new = np.concatenate((zDistBasis1, Zero), axis=1)

        print('New01: ')
        print(new)
        # zDistBasis1 = new
        zDistNew.append(new)

        add = np.concatenate((Zero, zDistBasis1), axis=1)
        zDistNew.append(add)


print('zDistNew: ')
zDist = np.array(zDistNew)
print(zDist)
