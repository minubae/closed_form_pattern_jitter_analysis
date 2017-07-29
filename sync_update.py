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

def getP_S(P_S):
    PS_T = np.array(P_S).T
    new = []
    for row in PS_T:
        new.append(np.sum(row))
    new = np.array(new)
    return new

givenT = getOmega(5, Target)
N = len(syncStateMat)
M = len(tDistMatrices)
where = 3
resutl = 0
# print('len tDistMatices: ', M)
P_S = []
P_S0 = []
P_S0_All = []

P_S1 = []
P_S1_All = []
P_S_Pre = []
P_S_Init = getInitSyncDist(initDist, syncStateMat)
P_S = getInitSyncDist(initDist, syncStateMat)
P_S_All = []
for k in range(where):

    print(syncStateMat[k+1])
    print(tDistMatrices[k])
    n = len(tDistMatrices[k])

    sDim = P_S.ndim

    P_S0_All = []
    P_S1_All = []

    # print('P_S_Pre: ')
    # print(P_S_Pre)
    #
    # print('New P_S: ')
    # print(P_S)

    for j in range(n):
        # print('Row: ', tDistMatrices[k][j])
        P_S0 = []
        P_S1 = []
        for i, prob_i in enumerate(tDistMatrices[k][j]):

            if syncStateMat[k+1][i] == 0:


                if sDim == 2:
                    # print('Non Sync: ', prob_i)
                    # print(syncStateMat[k+1][i])
                    result = prob_i*P_S[0][j]
                    # print('Result: ', result)
                    P_S0.append(result)
                    P_S1.append(0)
                    # print('Hello Yo')

                else:

                    # print('Hello 0')
                    # print('Pre: ', P_S_Pre[0][j])
                    # print(P_S[0][j][i])
                    result = prob_i*P_S[0][j][i] #P_S_Pre[0][j]
                    # print('result: ', result)
                    P_S0.append(result)
                    P_S1.append(0)

            else:

                if sDim == 2:
                    # print('Sync: ', prob_i)
                    # print(syncStateMat[k+1][i])
                    result = prob_i*P_S[1][j]
                    # print('Result: ', result)
                    P_S0.append(0)
                    P_S1.append(result)

                else:

                    # print('Hello 1')
                    # print('Pre: ', P_S_Pre[1][j][i])
                    # print(P_S[1][j][i])
                    result = prob_i*P_S[1][j][i] #*P_S_Pre[1][j][i]
                    # print('result: ', result)
                    P_S0.append(0)
                    P_S1.append(result)


        P_S0_All.append(P_S0)
        P_S1_All.append(P_S1)

    print('Hello There: ')
    print(np.array(P_S0_All))
    print(np.array(P_S1_All))

    nP_S0 = []
    nP_S1 = []
    nP_S0 = getP_S(P_S0_All)
    nP_S1 = getP_S(P_S1_All)


    P_S = []
    P_S.append(nP_S0)
    P_S.append(nP_S1)
    P_S = np.array(P_S)
    P_S_All.append(P_S)

    print('\n')

P_S1 = getInitSyncDist(initDist, syncStateMat)
print('P(S1): ')
print(P_S1)
print('P_S Matrix: ')
print(np.array(P_S_All), '\n')
