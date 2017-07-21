import numpy as np

# x = np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
# print('x: ', x)

# initDist: p(X_1)
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

tDistMatrices = np.array([
                         # p(X_2 | X_1)
                         [[5/22, 5/22, 5/22, 4/22, 3/22],
                          [0, 5/17, 5/17, 4/17, 3/17],
                          [0, 0, 5/12, 4/12, 3/12],
                          [0, 0, 0, 4/7, 3/7],
                          [0, 0, 0, 0, 1]],

                         # p(X_3 | X_2)
                         [[5/22, 5/22, 5/22, 4/22, 3/22],
                          [5/22, 5/22, 5/22, 4/22, 3/22],
                          [5/22, 5/22, 5/22, 4/22, 3/22],
                          [0, 5/17, 5/17, 4/17, 3/17],
                          [0, 0, 5/12, 4/12, 3/12]],

                         # p(X_4 | X_3)
                         [[5/15, 4/15, 2/15, 2/15, 1/15],
                          [5/15, 4/15, 2/15, 2/15, 1/15],
                          [5/15, 4/15, 2/15, 2/15, 1/15],
                          [0, 4/10, 3/10, 2/10, 1/10],
                          [0, 0, 3/6, 2/6, 1/6]],

                         # p(X_5 | X_4)
                         [[5/19, 5/19, 4/19, 3/19, 2/19],
                          [0, 5/14, 4/14, 3/14, 2/14],
                          [0, 0, 4/9, 3/9, 2/9],
                          [0, 0, 0, 3/5, 2/5],
                          [0, 0, 0, 0, 1]],

                         # p(X_6 | X_5)
                         [[5/15, 4/15, 2/15, 2/15, 1/15],
                          [5/15, 4/15, 2/15, 2/15, 1/15],
                          [0, 4/10, 3/10, 2/10, 1/10],
                          [0, 0, 3/6, 2/6, 1/6],
                          [0, 0, 0, 2/3, 1/3]],

                         # p(X_7 | X_6)
                         [[5/19, 5/19, 4/19, 3/19, 2/19],
                          [0, 5/14, 4/14, 3/14, 2/14],
                          [0, 0, 4/9, 3/9, 2/9],
                          [0, 0, 0, 3/5, 2/5],
                          [0, 0, 0, 0, 1]],

                         # p(X_8 | X_7)
                         [5/19, 5/19, 4/19, 3/19, 2/19]
                        ])

print('Transition Matrices: ')
print(tDistMatrices)

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

# print('X_i: ', getXi(tDistMat))
