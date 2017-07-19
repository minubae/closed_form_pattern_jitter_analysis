import numpy as np

# x = np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
# print('x: ', x)

tDist = [0.2, 0.2, 0.15, 0.20, 0.25]
def roll(dist):
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
        result+=1

def testRoll(N, transDist):
    result=[]
    for i in range(N):
        x=roll(transDist)
        print('X_1[',i,']:', x)
        result.append(x)
    return result

print(testRoll(10, tDist))


tDistMat = np.matrix([[0.20, 0.20, 0.15, 0.20, 0.25],
                      [0.15, 0.25, 0.10, 0.30, 0.20],
                      [0.25, 0.15, 0.20, 0.20, 0.20],
                      [0.10, 0.30, 0.20, 0.10, 0.30],
                      [0.30, 0.20, 0.15, 0.20, 0.15]])
# print('Transition Dist Matrix: ')
# print(tDistMat)
def getSurrogate(tDistMatrix):

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

# print('X_1: ', getSurrogate(tDistMat))
