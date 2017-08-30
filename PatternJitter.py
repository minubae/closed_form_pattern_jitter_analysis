###########################################################################################################################
# Title: Pattern Jitter Algorithm - Generating artificial spike trains
# Date: June/14/2017, Wednesday - August/25/2017, Friday
# Author: Minwoo Bae (minubae.math@gmail.com)
# Institute: Applied Mathematics, City College of New York, CUNY

# Abstract:
# Resampling methods are popular tools for exploring the statistical structure of neural spike trains.
# In many applications, it is desirable to have resamples that preserve certain non-Poisson properties,
# like refractory periods and bursting, and that are also robust to trial-to-trial variability.
# Pattern jitter is a resampling technique that accomplishes this by preserving the recent spiking history
# of all spikes and constraining resampled spikes to remain close to their original positions.
# The resampled spike times are maximally random up to these constraints. Dynamic programming is used to
# create an efficient resampling algorithm.
###########################################################################################################################
import numpy as np
import matplotlib.pyplot as plt
from random import *
# import itertools as itt

# Observed Spike Train
# x = np.random.uniform(0,1,(6,6))
# Generating a binary random spike train with size = n
#obs_x = np.random.randint(2, size=20)
#obs_x = np.array([0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,1,0,0])

""
# Finding a sequence of spike times from Observed splike data
# x_tilde = (x_tilde_1,..,x_tilde_n) denotes the Observed spike train,
# a non-decreasing sequence of spike times
# x_tilde: the observed spike train, nondecreasing sequence of spike times.
""

def getSpikeData(length, fireRate):

    T = 0
    coin = 0
    fRate = 0
    binprob = 0
    spikeData = []

    #length of train
    T = length;
    # initialize the Spike Train
    spikeData = np.zeros(T)
    fRate = fireRate
    binprob = (1./T)*fRate
    # print('binprob:', binprob)
    for k in range(0,int(T)):
        coin = np.random.uniform()
        # print('Coin:', coin)
        if coin <= binprob:
            spikeData[k] = 1

    return spikeData

# spikeData = getSpikeData(10, 4)
# print('Spike Data: ', spikeData)

def getSpikeTrain(spikeData):
    '''
    obs_x = observed_spike_train
    size = len(obs_x)
    x = []
    # Loop iteration with L-increments
    #for i in range(0, size, L): #print(x[i])
    for i in range(size):
        if obs_x[i] == 1:
            x.append(i+1)
    x_tilde = np.array(x)
    '''
    x_tilde = []
    x_tilde = spikeData
    x_tilde = np.where(spikeData==1)

    x_tilde = np.array(x_tilde)
    x_tilde = x_tilde.flatten()

    # print("Spike Data: ", spikeData)
    # print("Obs_x: ",x_tilde)
    # len(x_tilde)
    # print("Modified_Obs_x: ",x_tilde)

    return x_tilde

x_tilde = [10,15,20,26]
# x_tilde = getSpikeTrain(spikeData)
# print('Observed Spike Train: ', x_tilde)

def getX(spikeTrain):
    x = spikeTrain
    x = np.where(spikeTrain==1)
    x = np.array(x)
    x = x.flatten()
    return x

def getReference(Size, L, N):

    n = 0
    s = 0
    length = 0
    randInt = 0

    ref = []
    length = L

    s = Size+np.ceil(length/2)
    n = N

    for i in range(N):

        randIntTemp = np.random.randint(0, s) #np.sort(np.random.randint(0, s, size=n))
        # print('Yo: ', randIntTemp, N)
        if randInt < randIntTemp:
            ref.append(randIntTemp)
            randInt = randIntTemp
            randIntTemp = 0
        else:
            ref.append(randIntTemp)
            randInt = randIntTemp
            randIntTemp = 0


    ref = np.sort(ref)
    return ref

# x_tilde = get_x_tilde(get_spike_train(100))
# print('x_tilde: ', x_tilde)
# print('len_x_tilde: ', len(x_tilde), '\n')
# x = get_x(get_spike_train(100))

# print('X: ', x, '\n')
# print('len_X: ', len(x), '\n')

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

def getOmega(L, x_tilde):
    y = []
    n = len(x_tilde)

    for i in range(n):
        for j in range(1, L+1):
            y.append(x_tilde[i] - np.ceil(L/2) + j)
    Omega = np.array(y).reshape(n, L)

    return Omega

#Omega = getOmega(L, x_tilde)
#n = len(x_tilde)
#x = np.sort(np.random.randint(40, size=n))
# x_tilde = [10, 15, 22, 29, 34, 40, 45, 51]
# print(getOmega(5, x_tilde))

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

def getGamma(R, L, x_tilde):

    Gamma = []
    Gamma.append(0)
    n = len(x_tilde)

    for i in range(1, n):
        if x_tilde[i] - x_tilde[i-1] <= R:
            Gamma.append(x_tilde[i] - x_tilde[i-1])
        else:
            x = np.arange(R+1,R+1+L,1)
            Gamma.append(np.array(x))

    return Gamma

#Gamma = getGamma(R, x_tilde)

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
L = 3
R = 4
Omega = getOmega(L, x_tilde)
Gamma = getGamma(R, L, x_tilde)

# Indicator function 01 := 1{x[1] in Omega[1]}
def indicator_01(x_1):
    # numpy.in1d(ar1, ar2, assume_unique=False, invert=False)
    # Test whether each element of a 1-D array is also present in a second array.
    # Return a boolean array the same length as ar1 that is True where an element of ar1 is in ar2 and False otherwise
    # x = x_1
    if np.in1d(x_1, Omega[0]) == True:
    # if np.in1d(x[i], Omega[0]) == True:
        return 1

    return 0

# Indicator function 02 := 1{x[i] in Omega[i]}
def indicator_02(Xi, index):

    i = 0
    xi = 0

    xi = Xi
    i = index

    if np.in1d(xi, Omega[i]) == True:
        return 1
    return 0

# Indicator function 03 := 1{x[i] - (x[i]-1) in Gamma[i]}
def indicator_03(Xi_1, Xi, index):

    i = 0
    xi = 0
    xi_1 = 0

    xi = Xi
    xi_1 = Xi_1
    i = index

    if np.in1d(xi - xi_1, Gamma[i]) == True:
        # print('Gamma[',i+1,']: ', Gamma[i])
        # print('x[',i+1,'] - x[',i,']: ', x[i]-x[i-1])
        return 1

    # print('Gamma[',i+1,']: ', Gamma[i])
    # print('x[',i+1,'] - x[',i,']: ', x[i]-x[i-1])
    return 0

# h_1(x_1):= 1{x[1] in Omega[1]}
def h_1(x):
    # print('x: ', x)
    return indicator_01(x)

# h_i(x[i-1], x_i) := 1{x[i] in Omega[i]}*1{x[i]-x[i-1] in Gamma[i]}
def h_i(Xi_1, Xi, index):

    i = 0
    xi = 0
    xi_1 = []

    xi = Xi
    xi_1 = Xi_1

    i = index

    return indicator_02(xi, i)*indicator_03(xi_1, xi, i)

'''
Sampling from the Resampling Distribution
'''
# print('Omega_01:')
# print(Omega[0])
# print('Omega_02:')
# print(Omega[1])
# print(Omega[0][0])
# print('h_1 ? : ', h_1(x, 0),'\n')
# print('h_2 ? : ', h_i(x, 1))

print('x_tilde: ', x_tilde)
print('Omega:')
print(Omega)

print('Gamma:')
print(Gamma, '\n')
# print('Omega Length: ', len(Omega))


def Betai(Xi_1, Xi, index, ObsX):

    gap = 0
    betai = 0

    xi = Xi
    i = index
    xi_1 = Xi_1
    x_tilde = ObsX
    n = len(x_tilde)

    hi = hi = h_i(xi_1, xi, i)


    counter = i+1

    print('h_i: ', hi)
    print('x_tilde: ', x_tilde)
    print('x_tilde[i-1]: ', x_tilde[i-1])
    print('x_tilde[i]: ', x_tilde[i])
    # print('x_tilde[i] - x_tilde[i-1]: ', gap)
    print('\n')

    m = L

    if hi != 0:

        betai = 1
        for j in range(i+1, n):

            counter += 1

            print('index: ', j)
            print('xi: ', xi)
            print('Gamma i: ', Gamma[j])

            gap = x_tilde[j] - x_tilde[j-1]

            if np.array(Gamma[j-1]).ndim >= 1:

                print('case01')
                print('xi: ', xi)

                for k in range(m):

                    print('k: ', k)
                    print('xi[k]:', xi[k])

                    for s in range(j,n):
                        print('s: ', s)

                        xUp = xi[k]+Gamma[j]
                        print('xUp: ', xUp)

                # for x in xi:
                x_temp = xi+gap
                xi = x_temp
                print('x_temp: ', x_temp)

                if counter == n:

                    print('done1')

            else:

                print('case02')

                x_temp = xi+Gamma[j]
                xi = x_temp
                print('x_temp: ', x_temp)

                if counter == n:

                    print('done2')

            print('\n')

        return betai

    else:

        betai = 0
        return betai

# print(Betai(9, 12, 1, x_tilde))


def Beta1(inputX, X_tilde):

    x1 = 0
    h1 = 0
    count = 1
    x_tilde = []
    beta = []

    x1 = inputX
    h1 = h_1(x1)
    x_tilde = X_tilde
    # print('h1: ', h1)

    n = len(x_tilde)

    if h1 != 0:
        for i in range(1, n):

            count += 1
            x_temp = x1+Gamma[i]

            print('n: ', i)
            print('Gamma: ', Gamma[i])
            print('x[',i+1,'] temp: ', x_temp)
            print('Omega[',i+1,']: ', Omega[i])
            print('\n')

            if count == n:
                output = np.intersect1d(x_temp, Omega[i])
                beta.append(len(output))

            x1 = x_temp

        sumBeta = np.sum(beta)
        return sumBeta

    else:

        return 0

print('Beta1: ', Beta1(9, x_tilde))
# print('\n')


# L = 3
# R = 4
# x_tilde = [10, 13, 18, 23]
spikeX = [9, 12, 19, 23]

def Beta1P(X, L):
    m = 0
    n = 0
    x1_p = 0
    allOutput = []

    m = L
    n = len(X)

    for i in range(m):

        count = 1
        output = []
        x1_p = Omega[0][i]

        for j in range(1,n):
            count += 1
            x_temp = x1_p+Gamma[j]

            # print('n: ', j)
            # print('Gamma: ', Gamma[j])
            # print('x[',j+1,'] temp: ', x_temp)
            # print('Omega[',j+1,']: ', Omega[j])

            if count == n:
                output = np.intersect1d(x_temp, Omega[j])
                # print('Hey, Yo! How many outputs are : ', output)
                allOutput.append(len(output))

            x1_p = x_temp

    sumBetaP = np.sum(allOutput)
    return sumBetaP

# print('Beta1_Prime: ', Beta1P(X, L))


def p1(Omega, X, X_tilde):

    omega = []
    x_tilde = []
    initDist = []
    spikeX = []

    spikeX = X
    omega = Omega
    x_tilde = X_tilde

    print('Beta1_Prime: ', Beta1P(spikeX, L))
    print('\n')

    for x in omega[0]:
        print('x1: ', x)
        print('Beta1: ', Beta1(x, x_tilde))
        print('\n')
        initDist.append(Beta1(x, x_tilde) / Beta1P(spikeX, L))

    return initDist

# print('P1: ', p1(Omega, spikeX, x_tilde))
