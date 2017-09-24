###########################################################################################################################
# Title: Pattern Jitter Algorithm - Generating Artificial Spike Trains
# Date: June/14/2017, Wednesday - September/24/2017, Sunday
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

x_tilde = [10,13,18,22,26] #,26
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

def getGamma(R, L, Xtilde):

    n = 0
    gap = 0
    Gamma = []
    x_tilde = []

    Gamma.append(0)
    x_tilde = Xtilde
    n = len(x_tilde)


    for i in range(1, n):

        gap = x_tilde[i] - x_tilde[i-1]

        if gap <= R:
            Gamma.append(gap)
        else:
            r = R+1
            x = np.arange(r,r+gap,1)
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
R = 3
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
    xi_1 = 0

    xi = Xi
    xi_1 = Xi_1

    i = index

    # print('1: ', xi, indicator_02(xi, i))
    # print('2: ', xi_1, indicator_03(xi_1, xi, i))

    return indicator_02(xi, i)*indicator_03(xi_1, xi, i)


def hiVector(Xi_1, Xi, Index):

    X_i = Xi
    xi_1 = Xi_1
    output = []
    index = Index

    for i, xi in enumerate(X_i):

        hi = h_i(xi_1, xi ,index)
        output.append(hi)

    return output


'''
Sampling from the Resampling Distribution
'''
'''
print('x_tilde: ', x_tilde)
print('L: ', L, 'R: ', R)
# print('\n')
print('Omega:')
print(Omega)
print('Gamma:')
print(Gamma, '\n')
'''

def Beta1(X1, XTilde, Omega):

    m = 0; n = 0; h1 = 0; hi = 0
    index = 0; sumTemp = 0; beta1 = 0

    temp = []; betaTmp = []; hiSum = []
    omega = []; xi_1Tmp = []; vecTmp = []

    x1 = X1
    h1 = h_1(x1)
    omega = Omega
    xTilde = XTilde
    n = len(xTilde)

    if h1 == 1:

        index += 1
        Xi = omega[index]

        for i, xi in enumerate(Xi):

            hi = h_i(x1,xi,1)
            temp.append(hi)

            if hi == 1:
                xi_1Tmp.append(xi)
            else:
                xi_1Tmp.append(0)

        sumTemp = np.sum(temp)
        betaTmp.append(sumTemp)

        # print('xiTmp: ', xi_1Tmp)
        # print('\n')

        for i in range(2,n):

            temp = []
            index += 1
            sumTemp = 0

            Xi_1 = omega[index-1]
            Xi = omega[index]

            # print('index: ', index)
            # print('Xi_1: ', Xi_1)
            # print('Xi: ', Xi)

            for j, xi_1 in enumerate(xi_1Tmp):
                for k, xi in enumerate(Xi):

                    hi = h_i(xi_1, xi, index)
                    temp.append(hi)

                if xi_1 != 0:

                    vec = hiVector(xi_1, Xi, index)
                    vecTmp.append(vec)
                    # print('xi_1: ', xi_1)
                    # print('GetVecTmp: ', vecTmp)

            #vecTmp = []
            xi_1Tmp = []
            sumTemp = np.sum(temp)
            betaTmp.append(sumTemp)
            # vecTmp = np.array(vecTmp)
            # vecTmp = vecTmp.sum(axis=0)
            xi_1Tmp = vecTmp * np.array(Xi)
            xi_1Tmp = np.array(xi_1Tmp[0])

            # print('NewVecTmp: ', vecTmp)
            # print('New Xi_1 Temp: ', xi_1Tmp)
            # print('\n')

    else:

        beta1 = 0

    # print('betaTemp: ', betaTmp)
    beta1 = np.prod(betaTmp)

    return beta1


# print('Beta1: ', Beta1(9, x_tilde, Omega))
# x_tilde = [10,13,18,22]
# spikeX = [9, 14, 19, 23]

def Beta1P(Xtilde, Omega):

    beta1P = 0
    beta1Ptmp = []
    omega = Omega
    xTilde = Xtilde

    for i, x1_p in enumerate(omega[0]):
        beta1Ptmp.append(Beta1(x1_p, xTilde, omega))

    beta1P = np.sum(beta1Ptmp)
    return beta1P


def P1(X_tilde, binSize, historyLen):

    L = 0; R = 0; p1 = 0; beta1 = 0; beta1Prm = 0
    Omega = []; x_tilde = []; initDist = []

    L = binSize; R = historyLen
    x_tilde = X_tilde; Omega = getOmega(L, x_tilde)

    for i, x1 in enumerate(Omega[0]):

        beta1 = Beta1(x1, x_tilde, Omega)
        beta1Prm = Beta1P(x_tilde, Omega)
        p1 = beta1/beta1Prm

        initDist.append(p1)

        # print('x1: ', x1)
        # print('Beta1: ', Beta1(x1, xTilde, omega))
        # print('\n')

    # print('Beta1_Prime: ', Beta1P(xTilde, omega))
    # print('\n')

    initDist = np.array(initDist)
    return initDist

# print('P1: ', P1(Omega, x_tilde))
# print('Sum P1: ', np.sum(P1))

def Betai(Xi_1, Xi, Index, XTilde, Omega):

    n = 0; hi = 0; betai = 0; index = 0; sumTemp = 0

    temp = []; gear = []; betaTmp = []; hiSum = []
    omega = []; xi_1Tmp = []; vecTmp = []

    xi = Xi
    xi_1 = Xi_1
    omega = Omega
    index = Index
    xTilde = XTilde
    n = len(xTilde)

    # print('n: ', n)
    # print('index: ', index)

    hi = h_i(xi_1, xi, index)

    if hi == 1:

        if n-1 == index:

            # print('hello')
            Xi = omega[index]
            # print(xi_1)
            for i, xi in enumerate(Xi):
                # print(xi)
                hi = h_i(xi_1,xi,index)
                temp.append(hi)

                if hi == 1:
                    xi_1Tmp.append(xi)
                else:
                    xi_1Tmp.append(0)

            sumTemp = np.sum(temp)
            betaTmp.append(sumTemp)

        else:

            index += 1
            # print('index: ', index)

            xi_1 = xi
            Xi = omega[index]

            for i, xi in enumerate(Xi):

                hi = h_i(xi_1,xi,index)
                temp.append(hi)

                if hi == 1:
                    xi_1Tmp.append(xi)
                else:
                    xi_1Tmp.append(0)

            sumTemp = np.sum(temp)
            betaTmp.append(sumTemp)

            # print('temp: ', temp)
            # print('beta Temp: ', betaTmp)

            num = index+1

            for i in range(n-num):

                temp = []
                index += 1
                sumTemp = 0

                Xi_1 = omega[index-1]
                Xi = omega[index]

                # print('index: ', index)
                # print('xi_1: ', Xi_1)
                # print('xi: ', Xi)

                for j, xi_1 in enumerate(xi_1Tmp):
                    for k, xi in enumerate(Xi):

                        hi = h_i(xi_1, xi, index)
                        temp.append(hi)


                    if xi_1 != 0:

                        vec = hiVector(xi_1, Xi, index)
                        vecTmp.append(vec)

                xi_1Tmp = []
                sumTemp = np.sum(temp)
                betaTmp.append(sumTemp)

                xi_1Tmp = vecTmp * np.array(Xi)
                xi_1Tmp = np.array(xi_1Tmp[0])

    else:

        betai = 0

    # print('betaTmp: ', betaTmp)
    betai = np.prod(betaTmp)

    return betai

# print('Betai: ', Betai(9, 12, 1, x_tilde, Omega))
# print('Betai: ', Betai(12, 17, 2, x_tilde, Omega))
# print('Betai: ', Betai(17, 21, 3, x_tilde, Omega))
# print('Betai: ', Betai(21, 25, 4, x_tilde, Omega))

def Beta_iPrm(Xi_1, Index, Xtilde, Omega):

    # Initialize Variables
    xi_1 = 0; index = 0;
    betai = 0; beta_iPrm = 0

    temp = []; omega = []
    Xi = []; x_tilde = []

    # Set Variables
    xi_1 = Xi_1; index = Index; omega = Omega
    Xi = omega[index]; x_tilde = Xtilde

    # Iterate Xi to get the betai
    for i, xi in enumerate(Xi):

        betai = Betai(xi_1, xi, index, x_tilde, omega)
        temp.append(betai)

    beta_iPrm = np.sum(temp)

    return beta_iPrm

# print('Beta_iPrm: ', Beta_iPrm(12, 2, x_tilde, Omega))


def Pi(Xi_1, Xi, Index, X_tilde, Omega):

    xi_1 = 0; xi = 0
    index = 0; pi = 0
    x_tilde = []; omega = []

    xi_1 = Xi_1; xi = Xi
    index = Index; x_tilde = X_tilde
    omega = Omega

    betai = Betai(xi_1, xi, index, x_tilde, omega)
    beta_iPrm = Beta_iPrm(xi_1, index, x_tilde, omega)

    pi = betai/beta_iPrm

    return pi

# print('Pi: ', Pi(9, 12, 1, x_tilde, Omega))


def getTranMatrix(X_tilde, Index, Omega):

    index = 0; betai = 0
    beta_iPrm = 0; pi = 0

    omega = []; x_tilde = []
    Xi_1 = []; Xi = []
    temp = []; tranProb = []

    index = Index; omega = Omega
    x_tilde = X_tilde

    Xi_1 = omega[index-1]
    Xi = omega[index]

    for i, xi_1 in enumerate(Xi_1):
        for j, xi in enumerate(Xi):

            betai = Betai(xi_1, xi, index, x_tilde, omega)
            beta_iPrm = Beta_iPrm(xi_1, index, x_tilde, omega)
            pi = betai/beta_iPrm
            temp.append(pi)
            # print('Pi:', pi)

        # print('Sum: ', np.sum(temp))
        tranProb.append(temp)
        temp = []

    tranProb = np.array(tranProb)
    return tranProb

# print(getTranMatrix(1, x_tilde, Omega))


def getTranMatrices(X_tilde, binSize, historyLen):

    L = 0; R = 0; n = 0
    tranMatrix = []; tranMatrices = []
    Omega = []; x_tilde = []

    L = binSize; R = historyLen
    x_tilde = X_tilde; n = len(x_tilde)
    Omega = getOmega(L, x_tilde)

    for i in range(1, n):
        # print('i: ', i)
        tranMatrix = getTranMatrix(x_tilde, i, Omega)
        tranMatrices.append(tranMatrix)

    tranMatrices = np.array(tranMatrices)
    return tranMatrices

'''
initDist = P1(x_tilde, L, R)
tranMatrices = getTranMatrices(x_tilde, L, R)
print('\n')
print('######################################################')
print('Initial Distribution:')
print(initDist)
print('\n')

print('######################################################')
print('Transition Matrices:')
print(tranMatrices)
'''
