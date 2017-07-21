
import numpy as np
import matplotlib.pyplot as plt

R = [10, 15, 22, 29, 34, 40, 45, 51]
T = np.matrix([[8, 15, 20, 31, 34, 40, 43, 51],
               [10, 14, 21, 29, 34, 39, 46, 52],
               [11, 16, 22, 31, 34, 41, 45, 53],
               [12, 17, 23, 29, 34, 38, 45, 49],
               [19, 14, 24, 27, 34, 39, 44, 50],
               [9, 15, 22, 29, 33, 41, 47, 51],
               [12, 14, 21, 31, 34, 38, 45, 53],
               [11, 15, 22, 30, 34, 38, 46, 52],
               [10, 16, 24, 29, 32, 39, 43, 49],
               [8, 15, 22, 27, 34, 40, 44, 51],
               [9, 17, 20, 28, 32, 41, 46, 50],
               [11, 15, 22, 31, 33, 39, 44, 53],
               [10, 15, 21, 29, 34, 40, 43, 51],
               [10, 15, 22, 29, 33, 40, 46, 49],
               [10, 15, 22, 29, 34, 37, 45, 48],
               [9, 15, 22, 29, 34, 40, 45, 51],
               [10, 15, 22, 29, 34, 40, 45, 51]])

# print('Tmat:')
# print(T)

def getAmountSync(R, Tmat):
    s = 0
    S = []
    for j, row in enumerate(T):
        # Check how many elements are equal in two arrays (R, T)
        s = np.sum(R == row)
        S.append(s)
        # print(s)
    return S

S = getAmountSync(R, T)
print('Amount_Synchrony: ', S)
plt.hist(S, bins='auto')
plt.show()

'''
S = [0,1,2,3,4,5,6,3,4,3,5,6,3,4,3,4,5,6,3,0,1]
plt.hist(S, bins='auto')
plt.title("Histogram with 'auto' bins")
print('S: ', S)
plt.show()

x = np.random.normal(size = 1000)
plt.hist(x, normed=True, bins=30)
plt.ylabel('Probability');
plt.show()

def getAmountSync(R, Tmat):
    s = 0
    n = len(R)
    for i in range(n):
        if np.in1d(R[i], T[i]) == True:
            s+=1
    return s

def getAmountSyncTest(R, Tmat):
    s = 0
    for r in R:
        # print('r: ', r)
        for t in T:
            if r == t:
                s+=1
                break
    return s
'''
