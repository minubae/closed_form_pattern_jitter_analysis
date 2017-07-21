import numpy as np

# x = np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
# print('x: ', x)

# initDist = [0.2, 0.2, 0.15, 0.20, 0.25]
initDist = [5/15, 4/15, 3/15, 2/15, 1/15]
def getX1(dist):
    randX = np.random.random() # in [0,1)
    # print('Transition Dist.:', dist)
    # print('Random X:', randX)
    sum = 0
    result = 1
    i = 1
    for k in dist:
        # print('P[',i,']:', k)
        sum += k
        if randX <= sum:
            # print('Sum:', sum)
            return result
        i += 1
        result += 1

def testSampling(N, Dist):
    result=[]
    for i in range(N):
        x=getX1(Dist)
        print('X_1[',i,']:', x)
        result.append(x)
    return result

# print(testSampling(10, initDist))

'''
tDistMatrices = np.matrix([[0.20, 0.20, 0.15, 0.20, 0.25],
                      [0.15, 0.25, 0.10, 0.30, 0.20],
                      [0.25, 0.15, 0.20, 0.20, 0.20],
                      [0.10, 0.30, 0.20, 0.10, 0.30],
                      [0.30, 0.20, 0.15, 0.20, 0.15]],

                      [[0.20, 0.20, 0.15, 0.20, 0.25],
                       [0.15, 0.25, 0.10, 0.30, 0.20],
                       [0.25, 0.15, 0.20, 0.20, 0.20],
                       [0.10, 0.30, 0.20, 0.10, 0.30],
                       [0.30, 0.20, 0.15, 0.20, 0.15]])
'''

tDistMat = np.matrix([[0.20, 0.20, 0.15, 0.20, 0.25],
                      [0.15, 0.25, 0.10, 0.30, 0.20],
                      [0.25, 0.15, 0.20, 0.20, 0.20],
                      [0.10, 0.30, 0.20, 0.10, 0.30],
                      [0.30, 0.20, 0.15, 0.20, 0.15]])

# print('Transition Dist Matrix: ')
# print(tDistMat)
def getXi(tDistMatrix):

    sampleX = []
    n = len(tDistMatrix)

    for i, row in enumerate(tDistMat):
        print('tDistMat[',i,']:', row)
        rowDist = np.squeeze(np.array(row))
        sum = 0
        result = 1
        randX = np.random.random()
        m = len(rowDist)
        print('Rand_X:', randX)
        print('Sum:', sum)
        print('Result:', result)
        # print(rowDist)
        for j in range(m):
            print(j, rowDist[j])
            sum += rowDist[j]

            if randX <= sum:
                print('Sum:', sum)
                print('result: ', result, '\n')
                sampleX.append(result)
                break

            result+=1
    return sampleX

print('X_i: ', getXi(tDistMat))
