import numpy as np
from PatternJitter import *

x_tilde = [10, 15, 22, 29, 34, 40, 45, 51]
# x_tilde = [10, 15, 22, 29, 34, 40, 45, 51, 55]
# L = 5; R = 4

# initDist: p(X_1)
initDist = [5/15, 4/15, 3/15, 2/15, 1/15]

# p(X_2 | X_1)
p2 = np.array([[5/22, 5/22, 5/22, 4/22, 3/22],
               [0, 5/17, 5/17, 4/17, 3/17],
               [0, 0, 5/12, 4/12, 3/12],
               [0, 0, 0, 4/7, 3/7],
               [0, 0, 0, 0, 1]])
# p(X_3 | X_2)
p3 = np.array([[5/22, 5/22, 5/22, 4/22, 3/22],
               [5/22, 5/22, 5/22, 4/22, 3/22],
               [5/22, 5/22, 5/22, 4/22, 3/22],
               [0, 5/17, 5/17, 4/17, 3/17],
               [0, 0, 5/12, 4/12, 3/12]])

# p(X_4 | X_3)
p4 = np.array([[5/15, 4/15, 3/15, 2/15, 1/15],
               [5/15, 4/15, 3/15, 2/15, 1/15],
               [5/15, 4/15, 3/15, 2/15, 1/15],
               [0, 4/10, 3/10, 2/10, 1/10],
               [0, 0, 3/6, 2/6, 1/6]])

# p(X_5 | X_4)
p5 = np.array([[5/19, 5/19, 4/19, 3/19, 2/19],
               [0, 5/14, 4/14, 3/14, 2/14],
               [0, 0, 4/9, 3/9, 2/9],
               [0, 0, 0, 3/5, 2/5],
               [0, 0, 0, 0, 1]])

# p(X_6 | X_5)
p6 = np.array([[5/15, 4/15, 3/15, 2/15, 1/15],
               [5/15, 4/15, 3/15, 2/15, 1/15],
               [0, 4/10, 3/10, 2/10, 1/10],
               [0, 0, 3/6, 2/6, 1/6],
               [0, 0, 0, 2/3, 1/3]])

# p(X_7 | X_6)
p7 = np.array([[5/19, 5/19, 4/19, 3/19, 2/19],
               [0, 5/14, 4/14, 3/14, 2/14],
               [0, 0, 4/9, 3/9, 2/9],
               [0, 0, 0, 3/5, 2/5],
               [0, 0, 0, 0, 1]])
# p(X_8 | X_7)

p8 = np.array([[1/5, 1/5, 1/5, 1/5, 1/5],
               [1/5, 1/5, 1/5, 1/5, 1/5],
               [0, 1/4, 1/4, 1/4, 1/4],
               [0, 0, 1/3, 1/3, 1/3],
               [0, 0, 0, 1/2, 1/2]])
'''
p8 = np.array([[1/4, 1/4, 1/4, 1/4, 0],
               [1/4, 1/4, 1/4, 1/4, 0],
               [0, 1/3, 1/3, 1/3, 0],
               [0, 0, 1/2, 1/2, 0],
               [0, 0, 0, 1, 0]])

# p(X_9 | X_8)
p9 = np.array([[0, 1, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 1, 0],
               [0, 0, 0, 0, 1],
               [0, 0, 0, 0, 0]])
'''

tDistMatrices = np.stack((p2,p3,p4,p5,p6,p7,p8))

x_tilde_02 = [10, 14, 17]
x_tilde_03 = [10, 14, 17, 19]

# initDist: p(X_1)
initDist_02 = [1/3, 1/3, 1/3]
initDist_03 = [2/5, 2/5, 1/5]

# p(X_2 | X_1)
p2_02 = np.array([[1/3, 1/3, 1/3],
                  [1/3, 1/3, 1/3],
                  [1/3, 1/3, 1/3]])

p2_03 = np.array([[2/5, 2/5, 1/5],
                  [2/5, 2/5, 1/5],
                  [2/5, 2/5, 1/5]])


# p(X_3 | X_2)
p3_02 = np.array([[1/3, 1/3, 1/3],
                  [1/3, 1/3, 1/3],
                  [0, 1/2, 1/2]])

p3_03 = np.array([[3/6, 2/6, 1/6],
                  [3/6, 2/6, 1/6],
                  [0, 2/3, 1/3]])

# p(X_4 | X_3)

p4_03 = np.array([[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]])

tDistMatrices_02 = np.stack((p2_02, p3_02))

tDistMatrices_03 = np.stack((p2_03, p3_03, p4_03))


'''
a = [1,0,1]
a = np.array(a)
b = np.matmul(p2_03, a.T) # matmul
c = np.multiply(a, initDist_03)
print(b)
print(np.multiply(a, b))
'''

def Beta1(X1, XTilde, Omega):

    m = 0; n = 0
    h1 = 0; hi = 0; index = 0
    sumTemp = 0; beta1 = 0

    temp = []; betaTmp = []
    hiSum = []; omega = []

    x1 = X1
    h1 = h_1(x1)
    omega = Omega
    xTilde = XTilde
    n = len(xTilde)

    xi_1Tmp = []
    if h1 == 1:

        index += 1
        Xi = omega[index]

        print('index: ', index)

        for i, xi in enumerate(Xi):

            hi = h_i(x1,xi,1)
            temp.append(hi)

            if hi == 1:
                xi_1Tmp.append(xi)
            else:
                xi_1Tmp.append(0)

        sumTemp = np.sum(temp)
        betaTmp.append(sumTemp)

        print('xiTmp: ', xi_1Tmp)


        hi = 0
        index += 1
        Xi = omega[index]
        temp = []

        print('Xi: ', Xi)
        print('\n')

        xi_1Tmp2 = []

        for i, xi_1 in enumerate(xi_1Tmp):

            print('xi_1: ', xi_1)

            for j, xi in enumerate(Xi):

                hi = h_i(xi_1, xi,index)
                temp.append(h_i(xi_1,xi,index))

        sumTemp = np.sum(temp)
        betaTmp.append(sumTemp)

        print('\n')

        for i in range(3,n):

            temp = []
            index += 1
            sumTemp = 0

            Xi_1 = omega[index-1]
            Xi = omega[index]

            for j, xi_1 in enumerate(Xi_1):
                for k, xi in enumerate(Xi):

                    preXi_1 = omega[index-2][k]

                    print('Pre Xi-1: ', preXi_1)
                    print('Xi-1: ', xi_1)
                    print('Xi: ', xi)
                    preh_i = h_i(preXi_1, xi_1, index-1)
                    newh_i = h_i(xi_1, xi, index)

                    print('Pre H_i: ', preh_i)
                    print('New H_i: ', newh_i)
                    print('\n')


                    if preh_i*newh_i == 1:
                        temp.append(h_i(xi_1, xi, index))

            sumTemp = np.sum(temp)
            betaTmp.append(sumTemp)

    else:

        beta1 = 0

    print('betaTemp: ', betaTmp)
    beta1 = np.prod(betaTmp)

    return beta1


def Beta1(X1, XTilde, Omega):

    m = 0
    n = 0
    h1 = 0
    hi = 0
    beta1 = 0

    temp = []
    betaTmp = []
    hiSum = []
    omega = []

    x1 = X1
    h1 = h_1(x1)
    omega = Omega
    xTilde = XTilde
    n = len(xTilde)

    if h1 == 1:

        for i, Xi in enumerate(omega[1]):
            xi = Xi
            hi = h_i(x1,xi,1)
            temp.append(hi)

        h2_Sum = np.sum(temp)
        betaTmp.append(h2_Sum)

        # print('temp: ', 1, temp)
        # print('h2_Sum: ', h2_Sum)
        # print('\n')


        for i in range(2,n):

            hiSum = []
            # print('temp: ', i, temp)

            for j, hi_1 in enumerate(temp):

                if hi_1 == 1:

                    count = 0
                    temp = []
                    m = len(omega[i])
                    xi_1 = omega[i-1][j]

                    for k, Xi in enumerate(omega[i]):

                        xi = Xi
                        hi = h_i(xi_1, xi, i)
                        temp.append(hi)
                        count += 1

                        # print('index: ', i)
                        # print('xi_1: ', xi_1)
                        # print('xi: ', xi)
                        # print('hi: ', hi)
                        # print('temp: ', temp)
                        # print('\n')

                        if count == m:
                            hiSum.append(np.sum(temp))


            betaTmp.append(np.sum(hiSum))
            beta1 = np.prod(betaTmp)
            # print('beta tmp: ', i , betaTmp)
            # print('\n')

    else:

        beta1 = 0

    return beta1
