# Title: Pattern Jitter Algorithm - Generating artificial spike trains
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
# For each i = 1,...,n
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
# For each i = 2,...,n,
# X_{i} - X_{i-1} in Gamma_i where
# Gamma_i =
# {x_tilde_{i} - x_tilde_{i-1}} if x_tilde_{i} - x_tilde_{i-1} is less than or equal to R,
# or
# {R+1, R+2,...}  if x_tilde_{i} - x_tilde_{i-1} is greater than R.
# The parameter R controls the amount of history that is preserved. Larger values of R
# enforce more regularity in the firing patterns across the resampled spike trains.
R = 1
gamma = []
gamma.append(0)
for i in range(1, n):
    #print(x_tilde[i])
    if x_tilde[i] - x_tilde[i-1] <= R:
        print('hello 01')
        gamma.append(x_tilde[i] - x_tilde[i-1])
    else:
        x = np.arange(R,L,1)
        gamma.append(x)
        print('hello 02', gamma)
        #break

# numpy.arange(start, stop, step, dype=none)
# Return evenly spaced values within a given interval

# numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
# Return evenly spaced numbers over a specified interval

print('gamma:', gamma)
#print('Hello World!!')
print("Observed_X: ", obs_x)
print("x_tilde: ", x_tilde)
print("Omega: ")
print(Omega)

# Iterate over Omega matrix columnwise
#for i in Omega:
    #print(i)
    #for j in i:
        #print(j)
