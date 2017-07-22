import math
import numpy as np
import matplotlib.pyplot as plt
from progress import *

# x = np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
# print('x: ', x)
x_tilde = [10, 15, 22, 29, 34, 40, 45, 51]
L = 5; R = 4


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

def getSpikeTrain(obsX, L, R, initialDist, transMatrices):

    x1 = getX1(initDist, L, R, obsX)
    Omega = getOmega(L, obsX)

    x_i = x1
    chain = 1
    spikeTrain = []
    spikeTrain.append(x1)

    for i, row in enumerate(tDistMatrices):
        up = 0
        sum = 0
        result = Omega[chain][up]

        # print('i: ', i)
        # print('chain: ', chain)
        # print('x_i: ', x_i)
        index = np.where(np.array(Omega[i]) == x_i)[0]
        # print('x index: ', index)
        # print('init_result: ', result)

        tDistMat = row
        p_i = np.squeeze(np.array(tDistMat[index]))
        # print('p_i: ', p_i)

        m = len(p_i)
        Summation = np.sum(p_i)
        randX = np.random.random()
        # print('randX: ', randX)

        if Summation != 0:
            for j in range(m):
                # print('hello: ', j, p_i[j])
                sum += p_i[j]
                # print('Sum:', sum)
                if randX <= sum:
                    # print('result: ', result)
                    spikeTrain.append(result)
                    x_i = result
                    # print('ha_x_i: ', x_i)
                    chain += 1
                    break

                result = Omega[chain][up]
                up += 1
        else:

            p_i = tDistMat[0]
            m = len(p_i)
            randX = np.random.random()
            # print('randX: ', randX)
            for j in range(m):
                # print('p: ', p_i[j])
                sum += p_i[j]
                # print('Sum:', sum)

                if randX <= sum:
                    # print('result: ', result)
                    spikeTrain.append(result)
                    break

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

'''
X1 = []
for i in range(40):
    x1 = getX1(initDist, L, R, x_tilde)
    X1.append(x1)

# print(spikeTrain)
# plt.hist(X1, bins='auto')
# plt.show()
'''

'''
def testSampling(N, Dist):
    result=[]
    for i in range(N):
        x = getX1(Dist, R)
        print('X_1[',i,']:', x)
        result.append(x)
    return result

print(testSampling(10, initDist))
'''
'''
spikeTrain = []
x1 = getX1(initDist, L, R, x_tilde)
spikeTrain.append(x1)


Omega = getOmega(L, x_tilde)


x_i = x1
chain = 1
sampleX = []
sampleX.append(x1)

for i, row in enumerate(tDistMatrices):
    up = 0
    sum = 0
    result = Omega[chain][up]

    # print('i: ', i)

    print('chain: ', chain)
    print('x_i: ', x_i)
    index = np.where(np.array(Omega[i]) == x_i)[0]
    print('x index: ', index)
    print('init_result: ', result)

    tDistMat = row
    p_i = np.squeeze(np.array(tDistMat[index]))
    print('p_i: ', p_i)

    m = len(p_i)
    total = np.sum(p_i)
    randX = np.random.random()
    print('randX: ', randX)

    if total != 0:
        for j in range(m):
            # print('hello: ', j, p_i[j])
            sum += p_i[j]
            print('Sum:', sum)
            if randX <= sum:
                print('result: ', result)
                sampleX.append(result)
                x_i = result
                # print('ha_x_i: ', x_i)
                chain += 1
                break
            up += 1
            result = Omega[chain][up]
    else:

        p_i = tDistMat[0]
        m = len(p_i)
        randX = np.random.random()
        print('randX: ', randX)
        for j in range(m):
            print('p: ', p_i[j])
            sum += p_i[j]
            print('Sum:', sum)

            if randX <= sum:
                print('result: ', result)
                sampleX.append(result)
                break

    print('\n')

print('sampleX: ', sampleX)
'''
