# Title: Pattern Jitter Algorithm - Generatin artificial spike trains
# Date: June/14/2017, Wednesday - Current
# Author: Minwoo Bae (minubae.math@gmail.com)
# Institute: Mathematics, City College of New York, CUNY

import numpy as np
# import itertools as itt

# Observed Spike Train
# x = np.random.uniform(0,1,(6,6))
# Generating a binary random spike train with size = n
obs_x = np.random.randint(2, size=20)
#obs_x = np.array([0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0])
size = len(obs_x)
L = 5

# Loop iteration with L-increments
#for i in range(0, size, L): #print(x[i])

# The Observed spike train, a non-decreasing sequence of spike times
x = []
for i in range(size):
    if obs_x[i] == 1:
        x.append(i+1)

# x_tilde: the observed spike train, nondecreasing sequence of spike times.
x_tilde = np.array(x)

# Jitter Window
y = []
n = len(x_tilde)
for i in range(n):
    for j in range(1, L+1):
        y.append(x_tilde[i] - np.ceil(L/2) + j)

Omega = np.array(y).reshape(n, L)

#print('Hello World!!')
print("Observed_X: ", obs_x)
print("x_tilde: ", x_tilde)
print("Omega: ")
print(Omega)

# Iterate over Omega matrix columnwise
for i in Omega:
    print(i)
    for j in i:
        print(j)
