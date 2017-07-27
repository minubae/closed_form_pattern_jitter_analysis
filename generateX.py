import numpy as np
import matplotlib.pyplot as plt
from progress import *
from Data import *

# x = np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
# print('x: ', x)

def getX1(dist, L, R, x_tilde):
    randX = np.random.random() # in [0,1)
    # print('Transition Dist.:', dist)
    # print('Random X:', randX)
    up = 0
    sum = 0
    Omega = getOmega(L, x_tilde)
    result = Omega[0][up]

    for k in dist:
        # print('P[',i,']:', k)
        sum += k
        if randX <= sum:
            # print('Sum:', sum)
            return result
        up += 1
        result = Omega[0][up]

def initializeX(initX, P):
    m = len(P)
    for k in range(m):
        if P[k] == 0:
            initX += 1
    return initX


# x_tilde = [10, 15, 22, 29, 34, 40, 45, 51]
# L = 5
# R = 4
def getSpikeTrain(obsX, L, R, initialDist, transMatrices):

    print('////**** Simulation is starting. ****////')

    x1 = getX1(initDist, L, R, obsX)
    Omega = getOmega(L, obsX)

    chain = 1
    givenX = x1
    spikeTrain = []
    spikeTrain.append(x1)

    for i, row in enumerate(tDistMatrices):

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

        if Summation != 0:

            initX = initializeX(Omega[chain][0], p_i)
            print('init_X: ', initX)

            randX = np.random.random()
            print('Roll a random X: ', randX)

            m = len(p_i)
            for j in range(m):

                if p_i[j] != 0:

                    # print('Check Sum: ', sum)

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
        else:

            p_i = tDistMat[0]
            m = len(p_i)
            initX = initializeX(Omega[chain][0], p_i)
            randX = np.random.random()
            print('Sorry, find an another transition probability.')
            print('p_i: ', p_i)
            print('init_X: ', initX)
            print('Roll a random X: ', randX)

            for j in range(m):

                sum += p_i[j]
                print('Sum:', sum)

                if randX <= sum:
                    print('Output X: ', initX)
                    print('Omega: ', Omega[chain])
                    if np.in1d(initX, Omega[chain]) == True:
                        spikeTrain.append(initX)
                    else:
                        spikeTrain.append(0)
                    break
                initX += 1

    print('////**** Simulation is done. ****////', '\n')
    return spikeTrain

# print(getSpikeTrain(x_tilde, 5, 4, initDist, tDistMatrices))


def getXi(tDistMatrix):

    sampleX = []
    n = len(tDistMatrix)
    print(tDistMatrix)
    for i, row in enumerate(tDistMatrix):
        print('tDistMat[',i,']:', row)
        p_i = np.squeeze(np.array(row))
        print('p_i: ', p_i)
        sum = 0
        result = 1
        randX = np.random.random()
        m = len(p_i)
        print('Rand_X:', randX)
        print('Sum:', sum)
        print('Result:', result)
        # print(rowDist)
        for j in range(m):
            print(j, p_i[j])
            sum += p_i[j]

            if randX <= sum:
                print('Sum:', sum)
                print('result: ', result, '\n')
                sampleX.append(result)
                break

            result+=1
    return sampleX

# print('X_i: ', getXi(tDistMat))
