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
# print('Sync State Matrix: ')
# print(syncStateMat, '\n')

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

def getNewP_S(P_S):
    PS_T = np.array(P_S).T
    new = []
    for row in PS_T:
        new.append(np.sum(row))
    new = np.array(new)
    return new

where = 1
result0 = 0
result1 = 0

P_S = []
P_S0 = []
P_S0_All = []

P_S1 = []
P_S1_All = []

P_S = getInitSyncDist(initDist, syncStateMat)
P_S_All = []
# P_S_All.append(P_S)
sumP_S = []
sumP_S_All = []
# sumP_S.append(np.sum(P_S[0]))
# sumP_S.append(np.sum(P_S[1]))
# sumP_S_All.append(np.array(sumP_S))
print('P_S1: ')
print(P_S, '\n')
ps1temp = []
ps0temp = []
for k in range(where):

    # print(syncStateMat[k+1])
    print(tDistMatrices[k], '\n')
    n = len(tDistMatrices[k])

    P_S0_All = []
    P_S1_All = []

    for j in range(n):
        # print('Row: ', tDistMatrices[k][j])
        P_S0 = []
        P_S1 = []
        temp0 = []
        temp1 = []

        for i, prob_i in enumerate(tDistMatrices[k][j]):

            if syncStateMat[k+1][i] == 0:

                # print('Non Sync: ', prob_i)
                # print(syncStateMat[k+1][i])

                # 0 -> 0
                result = prob_i*P_S[0][j]

                # 1 -> 1
                result1 = prob_i*P_S[1][j]
                # print('0: Result: ', result1)

                P_S0.append(result)
                P_S1.append(0)


                temp0.append(0)
                temp1.append(result1)

            else:

                # print('Sync: ', prob_i)
                # print(syncStateMat[k+1][i])

                temp1.append(0)


                # 1-> 2
                result = prob_i*P_S[1][j]

                # 0 -> 1
                result1 = prob_i*P_S[0][j]
                # print('1: Result: ', result1)

                P_S0.append(0)
                P_S1.append(result)
                temp0.append(result1)



        P_S0_All.append(P_S0)
        # print('0 -> 1', temp0)
        P_S0_All.append(temp0)
        P_S1_All.append(P_S1)

        ps0temp.append(temp0)
        ps1temp.append(temp1)

        P_S1_All.append(temp1)

    # print('P_S: ')
    # print(np.array(P_S0_All))
    # print(np.array(P_S1_All))


    nP_S0 = []
    nP_S1 = []
    sumP_S = []
    nP_S0 = getNewP_S(P_S0_All)
    nP_S1 = getNewP_S(P_S1_All)

    # print(nP_S0)
    # print(nP_S1)
    sumP_S0 = np.sum(nP_S0)
    sumP_S1 = np.sum(nP_S1)
    # print('Sum0: ', sumP_S0)
    # print('Sum1: ', sumP_S1)

    P_S = []
    P_S.append(nP_S0)
    P_S.append(nP_S1)
    P_S = np.array(P_S)
    P_S_All.append(P_S)

    sumP_S.append(sumP_S0)
    sumP_S.append(sumP_S1)
    sumP_S_All.append(sumP_S)

    print('\n')

print('0 -> 1')
print(np.array(ps0temp))

print('1 -> 1')
print(np.array(ps1temp))

'''
print('P_S Matrix: ')
print(np.array(P_S_All))
print('Sum P_S: ')
print(np.array(sumP_S_All))
print('\n')
'''
