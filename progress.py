###########################################################################################################################
# Title: Pattern Jitter Algorithm - Generating artificial spike trains
# Date: June/14/2017, Wednesday - Current
# Author: Minwoo Bae (minubae.math@gmail.com)
# Institute: Mathematics, City College of New York, CUNY

# Abstract:
# Resampling methods are popular tools for exploring the statistical structure of neural spike trains.
# In many applications, it is desirable to have resamples that preserve certain non-Poisson properties,
# like refractory periods and bursting, and that are also robust to trial-to-trial variability.
# Pattern jitter is a resampling technique that accomplishes this by preserving the recent spiking history
# of all spikes and constraining resampled spikes to remain close to their original positions.
# The resampled spike times are maximally random up to these constraints. Dynamic programming is used to
# create an efficient resampling algorithm.
###########################################################################################################################

from random import *
import numpy as np
import matplotlib.pyplot as plt
# import itertools as itt

# Observed Spike Train
# x = np.random.uniform(0,1,(6,6))
# Generating a binary random spike train with size = n
obs_x = np.random.randint(2, size=20)
#obs_x = np.array([0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0])

""
# Finding a sequence of spike times from Observed splike data
# x_tilde = (x_tilde_1,..,x_tilde_n) denotes the Observed spike train,
# a non-decreasing sequence of spike times
# x_tilde: the observed spike train, nondecreasing sequence of spike times.
""
def get_x_tilde(observed_spike_train):
    size = len(obs_x)
    x = []
    # Loop iteration with L-increments
    #for i in range(0, size, L): #print(x[i])
    for i in range(size):
        if obs_x[i] == 1:
            x.append(i+1)
    x_tilde = np.array(x)

    return x_tilde

x_tilde = get_x_tilde(obs_x)

""
# Preserving smoothed firing rates: we require that each resampled spike remain
# close to its corresponding original spike.
# Omega_i: the ith Jitter Window
# (2.1) For each i = 1,...,n
# X_i in Omega_i where Omega_i = {x_tilde_i - ceil(L/2)+1,...,x_tilde_i - ceil(L/2)+L}
# The parameter L controls the degree of smoothing: small L preserves rapid changes
# in firing rate but introduces less variability into resamples.
# L : the size of window
""
L = 5
def getOmega(L, x_tilde):
    y = []
    n = len(x_tilde)

    for i in range(n):
        for j in range(1, L+1):
            y.append(x_tilde[i] - np.ceil(L/2) + j)
    Omega = np.array(y).reshape(n, L)

    return Omega

Omega = getOmega(L, x_tilde)


n = len(x_tilde)
x = np.sort(np.random.randint(40, size=n))

""
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
""
R = 2
def getGamma(R, x_tilde):
    Gamma = []
    Gamma.append(0)
    for i in range(1, n):
        if x_tilde[i] - x_tilde[i-1] <= R:
            Gamma.append(x_tilde[i] - x_tilde[i-1])
        else:
            x = np.arange(R+1,R+1+L,1)
            Gamma.append(x)
    return Gamma

Gamma = getGamma(R, x_tilde)

""
# To the extent that an observed spike train conforms to such a model, the resampling distribution
# will preserve the essential history-dependent features of the model.
# There are many distributions that preserve (2.1) and (2.2). Since our goal is to improve no additional
# structure, we make no additional constraints: the allowable spike configurations are distributed
# uniformly, meaning that
# p(x) = 1/Z 1{x_1 in Omega_1} Product{from i =1 to n}1{x_i in Omega_i}1{x_i - x_{i-1} in Gamma_i},
# where 1{A} is the indicator function of the set A and Z is a normalization constant that depends on
# the Omega_i's and the Gamma_i's, and hence on the parameters L and R and the original spike train, x_tilde.
""

# Indicator function 01 := 1{x[1] in Omega[1]}
def indicator_01(x_1):
    # numpy.in1d(ar1, ar2, assume_unique=False, invert=False)
    # Test whether each element of a 1-D array is also present in a second array.
    # Return a boolean array the same length as ar1 that is True where an element of ar1 is in ar2 and False otherwise
    if np.in1d(x_1, Omega[0]) == True:
        return 1
    return 0

# Indicator function 02 := 1{x[i] in Omega[i]}
def indicator_02(i):
    print('Omega[',i+1,']: ', Omega[i])
    print('x[',i+1,']: ', x[i])
    if np.in1d(x[i], Omega[i]) == True:
        return 1
    return 0

# Indicator function 03 := 1{x[i] - (x[i]-1) in Gamma[i]}
def indicator_03(i):
    if np.in1d(x[i]-x[i-1], Gamma[i]) == True:
        print('Gamma[',i+1,']: ', Gamma[i])
        print('x[',i+1,'] - x[',i,']: ', x[i]-x[i-1])
        return 1

    print('Gamma[',i+1,']: ', Gamma[i])
    print('x[',i+1,'] - x[',i,']: ', x[i]-x[i-1])
    return 0

# Resampling Distribution p(x), where x = (x_1,...,x_n)
# p(x) := (1/Z)*h_1(x_1)Product{from i=2 to n}*h_i(x[i-1], x[i])
def p(Z, i): # Z
    return (1/Z)*h_1(x[0])*h_i(i)

# h_1(x_1) := Indicator function 01
def h_1(x_1):
    return indicator_01(x_1)

# h_i(x[i-1], x_i) := 1{x[i] in Omega[i]}*1{x[i]-x[i-1] in Gamma[i]}
def h_i(i):
    # print('Input: ', x_1, x_2)
    print('X: ', x)
    return indicator_02(i)*indicator_03(i)

t = np.sort(np.random.randint(40, size=n+1))

def p1(Z):
    return 1/Z

# Resampled Spike Train
X = []
# Sampling from the Resampling Distribution
# Acceptance-Rejection Algorithm to have the Sampling
def rejection_sampling():

    Z = 1
    # N is positive infinity
    N = float('inf')
    x_1 = randint(1,100)

    while x_1 < N:

        x_1 = randint(1,100)
        print('x_1: ', x_1)
        print('Omega[1]:', Omega[0], '\n')
        print('h_1: ', h_1(x_1))
        if h_1(x_1):
            X.append(x_1)
            return p1(Z)
            break

        Z = Z + 1
        # print('Counter: ', counter)


print('Distribution: ', rejection_sampling())
print('X: ', X)

"""
print("Observed_X: ", obs_x)
print("spike_time_observed_x: ", x_tilde)
print('spike_time_sampling_x: ', x, '\n')

print("Omega: ")
print(Omega)
print('Gamma:')
print(Gamma, '\n')

print('X: ', x)
print("Omega[1]: ", Omega[0])
print('x[1]: ', x[0])
print("Exist?: ", h_1(x[0]), '\n')

for i in range(1, n):
    print('Exist?: ', p(1, i), '\n')
"""

# y = np.zeros(N)
y = np.ones(n)
m = len(obs_x)
plt.plot(x_tilde, y, 'o')
plt.xlim([0, m])
# plt.ylim([-1, 1])
# plt.axis([0, m, -1, 1])
# plt.show()

#print('Hello World!!')
# print(Omega[:1,])
# print(Omega[0:1,])

# def isinteger(x):
#     print('Integer')
#     return np.equal(np.mod(x, 1), 0)
#
# def isarray(vector):
#     print('Array')
#     if isinteger(vector):
#         return False
#     else:
#         n = len(vector)
#         if n > 1:
#             return True

# indicator = lambda x_i, Omega_i: 1 if x_i == Omega_i else 0
# print(indicator(1,1))

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
# N = 8
# y = np.zeros(N)
# x1 = np.linspace(0, 10, N, endpoint=True)
# x2 = np.linspace(0, 10, N, endpoint=False)
# plt.plot(x1, y, 'o')
# plt.plot(x2, y + 0.5, 'o')
# plt.ylim([-0.5, 1])
# plt.show()
