import numpy as np
import matplotlib.pyplot as plt


S = [0,1,2,3,4,5,6,3,4,3,5,6,3,4,3,4,5,6,3,0,1]
plt.hist(S, bins='auto')
# plt.title("Histogram with 'auto' bins")
# print('S: ', S)
# plt.show()


'''
x = np.random.normal(size = 1000)
plt.hist(x, normed=True, bins=30)
plt.ylabel('Probability');
plt.show()
'''

R = [10, 15, 22, 29, 33, 37, 42, 48]
T = [8, 15, 20, 32, 33, 34, 45, 48]

s = 0
for r in R:
    # print('r: ', r)
    for t in T:
        if r == t:
            s+=1
            break
# print('s:', s)

a = 0
n = len(R)
for i in range(n):

    if np.in1d(R[i], T[i]) == True:
        a+=1
print('a: ', a)
