import numpy as np
from PatternJitter import *
from Data import *

Ref = [8, 18, 23, 28, 34, 42, 44, 49]

def getSyncState(L, Reference, Target):
    ref = Reference
    givenT = getOmega(L, Target)
    print('Reference Train: ', ref)
    # print('Given T: ')

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
                print('Check[', j,']: ', checkSync)
                print('Tj[',i,']: ', Tj[i], 'is SYNCRONY.')
                syncState.append(1)
            else:
                # print('Check[', j,']: ', checkSync)
                # print('Tj:', Tj[i], 'is not Synch.')
                syncState.append(0)

        syncStateMat.append(syncState)

    syncStateMat = np.array(syncStateMat)
    return syncStateMat

print(getSyncState(5, Ref, x_tilde))
print('Transition Matrix: ')
print(tDistMatrices)
