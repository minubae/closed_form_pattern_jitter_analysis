import numpy as np
from PatternJitter import *

def getInitDist(L):

    dist = []
    length = 0
    initDist = []

    length = L
    dist = np.random.rand(1,length)
    initDist = dist/dist.sum(axis=1)[:,None]

    initDist = np.squeeze(np.array(initDist))

    return initDist



def getTransitionMatrices(L, N):

    size = 0
    length = 0

    matrix = []
    tDistMatrices = []

    size = N
    length = L

    for i in range(size-1):
        # print(i)
        matrix = np.random.rand(length,length)
        stochMatrix = matrix/matrix.sum(axis=1)[:,None]
        tDistMatrices.append(stochMatrix.astype('f'))

    tDistMatrices = np.array(tDistMatrices)

    return tDistMatrices


'''
fRate = 6
Size = 20
spikeData = getSpikeData(Size, fRate)
spikeTrain = getSpikeTrain(spikeData)


print('Spike Data: ')
print(spikeData)
print('Spike Train: ')
print(spikeTrain)

n = len(spikeTrain)

L = 3
N = n
initDist = getInitDist(L)
tDistMatrices = getTransitionMatrices(L, N)

print('Initial Distribution: ')
print(initDist)
print('Transition Matrices: ')
print(tDistMatrices)
'''


'''
print('Sum of row: ', np.sum(initDist))
for k, row in enumerate(tDistMatrices[1]):
    print('Sum of row: ', k, np.sum(row))
'''

'''
import random
precision = 100000

def f(n) :
    matrix = []
    for l in range(n) :
        lineLst = []
        sum = 0
        crtPrec = precision
        for i in range(n-1) :
            val = random.randrange(crtPrec)
            sum += val
            lineLst.append(float(val)/precision)
            crtPrec -= val
        lineLst.append(float(precision - sum)/precision)
        matrix.append(lineLst)
    return matrix


matrix = f(3)
matrix = np.array(matrix)
print(matrix)
'''
