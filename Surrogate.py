import numpy as np
import matplotlib.pyplot as plt
from PatternJitter import *
from Data import *

# x = np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
# print('x: ', x)

def getX1(Dist, L, R, ObsX):
     # in [0,1)
    # print('Transition Dist.:', dist)
    # print('Random X:', randX)
    up = 0
    sum = 0
    length = 0
    hisLen = 0

    dist = []
    obsTar = []

    length = L
    hisLen = R
    dist = Dist
    obsTar = ObsX

    Omega = getOmega(length, obsTar)
    result = Omega[0][up]
    randX = np.random.random()

    for k in dist:
        # print('P[',i,']:', k)
        sum += k
        if randX <= sum:
            # print('Sum:', sum)
            return result
        up += 1
        result = Omega[0][up]

def initializeX(initX, Prob):

    init_x = 0
    prob = []

    init_x = initX
    prob = Prob
    m = len(prob)

    for k in range(m):
        if prob[k] == 0:
            init_x += 1

    return init_x

# getSpikeTrain(ObsTar, length, hisLen, initD, tDistMat)

def getSurrogate(obsX, L, R, initDist, tDistMatrices):

    print('////**** Simulation is starting. ****////')

    chain = 1
    length = 0
    hisLen = 0

    initD = []
    givenX = []
    obsTar = []
    tDistMat = []
    spikeTrain = []

    length = L
    hisLen = R
    obsTar = obsX
    initD = initDist
    tDistMat = tDistMatrices

    Omega = getOmega(length, obsTar)
    x1 = getX1(initD, length, hisLen, obsTar)

    givenX = x1
    spikeTrain.append(x1)

    for i, row in enumerate(tDistMat):

        sum = 0
        randX = 0
        index = np.where(np.array(Omega[i]) == givenX)[0]

        print('Chain: ', chain)
        print('Given X: ', givenX)
        print('Matrix row index: ', index)

        tDistMat = row
        p_i = np.squeeze(np.array(tDistMat[index]))
        print('p_i: ', p_i)

        Summation = np.sum(p_i)

        initX = initializeX(Omega[chain][0], p_i)
        print('init_X: ', initX)

        randX = np.random.random()
        print('Roll a random X: ', randX)

        m = len(p_i)

        for j in range(m):

            if p_i[j] != 0:

                sum += p_i[j]
                print('Sum:', sum)

                if randX <= sum:
                    print('Output X: ', initX)
                    spikeTrain.append(initX)
                    givenX = initX
                    print('New Given X: ', initX, '\n')
                    chain += 1
                    break

                print('Check InitX 02: ', initX)
                initX += 1

            else:
                j += 1

    print('////**** Simulation is done. ****////', '\n')
    return spikeTrain

# print(getSpikeTrain(x_tilde, 5, 4, initDist, tDistMatrices))
