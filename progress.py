# Title: Pattern Jitter Algorithm - Generating artificial spike trains
# Date: June/14/2017, Wednesday - Current
# Author: Minwoo Bae (minubae.math@gmail.com)
# Institute: Mathematics, City College of New York, CUNY

import numpy as np
import matplotlib.pyplot as plt
# import itertools as itt

# Observed Spike Train
# x = np.random.uniform(0,1,(6,6))
# Generating a binary random spike train with size = n
obs_x = np.random.randint(2, size=20)
#obs_x = np.array([0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0])
size = len(obs_x)

# Loop iteration with L-increments
#for i in range(0, size, L): #print(x[i])

# Finding a sequence of spike times from Observed splike data
# x_tilde = (x_tilde_1,..,x_tilde_n) denotes the Observed spike train,
# a non-decreasing sequence of spike times
x = []
for i in range(size):
    if obs_x[i] == 1:
        x.append(i+1)

# x_tilde: the observed spike train, nondecreasing sequence of spike times.
x_tilde = np.array(x)

# Preserving smoothed firing rates: we require that each resampled spike remain
# close to its corresponding original spike.
# Omega_i: the ith Jitter Window
# (2.1) For each i = 1,...,n
# X_i in Omega_i where Omega_i = {x_tilde_i - ceil(L/2)+1,...,x_tilde_i - ceil(L/2)+L}
# The parameter L controls the degree of smoothing: small L preserves rapid changes
# in firing rate but introduces less variability into resamples.
L = 5
y = []
n = len(x_tilde)
for i in range(n):
    for j in range(1, L+1):
        y.append(x_tilde[i] - np.ceil(L/2) + j)

Omega = np.array(y).reshape(n, L)

# Preserving recent spike history effects: we require that the resampled and
# the original recording have identical patterns of spiking and not spiking
# in the R bins preceding each spike.
# (2.2) For each i = 2,...,n,
# X_{i} - X_{i-1} in Gamma_i where
# Gamma_i =
# {x_tilde_{i} - x_tilde_{i-1}} if x_tilde_{i} - x_tilde_{i-1} is less than or equal to R,
# or
# {R+1, R+2,...}  if x_tilde_{i} - x_tilde_{i-1} is greater than R.
# The parameter R controls the amount of history that is preserved. Larger values of R
# enforce more regularity in the firing patterns across the resampled spike trains.
R = 2
Gamma = []
Gamma.append(0)
for i in range(1, n):
    if x_tilde[i] - x_tilde[i-1] <= R:
        Gamma.append(x_tilde[i] - x_tilde[i-1])
    else:
        x = np.arange(R+1,R+1+L,1)
        Gamma.append(x)

# To the extent that an observed spike train conforms to such a model, the resampling distribution
# will preserve the essential history-dependent features of the model.
# There are many distributions that preserve (2.1) and (2.2). Since our goal is to improve no additional
# structure, we make no additional constraints: the allowable spike configurations are distributed
# uniformly, meaning that
# p(x) = 1/Z 1{x_1 in Omega_1} Product{from i =1 to n}1{x_i in Omega_i}1{x_i - x_{i-1} in Gamma_i},
# where 1{A} is the indicator function of the set A and Z is a normalization constant that depends on
# the Omega_i's and the Gamma_i's, and hence on the parameters L and R and the original spike train, x_tilde.

# Resampling Distribution p(x), where x = (x_1,...,x_n)
n = len(Gamma)
x = np.sort(np.random.randint(40, size=n))
print(x)
def p(x):
    return False

def h_1(x):
    for i in Omega[0]:
        if i == x[0]:
            return 1
            break
    return 0

def h_i(x):
    return False

#print('Hello World!!')
print("Observed_X: ", obs_x)
print("x_tilde: ", x_tilde)
print("Omega: ")
print(Omega)
# print(Omega[:1,])
# print(Omega[0:1,])
print("Omega_[0]: ", Omega[0])
print("h_1: ", h_1(x))
print('Gamma:')
print(Gamma)

# for i in Gamma:
#     print('Gamma:', i)

# Iterate over Omega matrix columnwise
#for i in Omega:
    #print(i)
    #for j in i:
        #print(j)

# numpy.arange(start, stop, step, dype=none)
# Return evenly spaced values within a given interval

# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# Return evenly spaced numbers over a specified interval
# Testing Matplotlib in Scipy with linspace
N = 8
y = np.zeros(N)
x1 = np.linspace(0, 10, N, endpoint=True)
x2 = np.linspace(0, 10, N, endpoint=False)
plt.plot(x1, y, 'o')
plt.plot(x2, y + 0.5, 'o')
plt.ylim([-0.5, 1])
#plt.show()
