
import numpy as np
import matplotlib.pyplot as plt

R = [10, 15, 22, 29, 33, 37, 42, 48]
T = np.matrix([[8, 15, 20, 31, 33, 35, 42, 48],
               [10, 14, 21, 29, 33, 36, 40, 46],
               [11, 16, 22, 31, 32, 37, 41, 47],
               [12, 17, 23, 30, 33, 38, 43, 48],
               [13, 14, 24, 27, 34, 39, 44, 49],
               [9, 15, 22, 29, 33, 35, 42, 50],
               [12, 14, 21, 31, 33, 37, 43, 46],
               [11, 15, 22, 30, 34, 36, 40, 48],
               [10, 16, 24, 29, 32, 39, 41, 47],
               [8, 13, 23, 27, 31, 38, 42, 46],
               [9, 17, 20, 28, 32, 37, 43, 50],
               [11, 15, 22, 31, 33, 36, 44, 49],
               [10, 15, 21, 29, 33, 35, 42, 48],
               [10, 15, 22, 29, 33, 37, 42, 48],
               [8, 15, 24, 29, 33, 37, 42, 48],
               [9, 15, 20, 29, 33, 37, 40, 49],
               [11, 15, 22, 28, 32, 39, 41, 48]])

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
