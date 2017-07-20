'''
Case 01:
x_tilde[i] - x_tilde[i-1] > R == 0 ;
x_tilde[i] - x_tilde[i-1] <= L ;
x_tilde[i] - x_tilde[i-1] > Gamma[i][0] // first element of Gamma[i]
'''
beta = L
beta_prime = L*L

'''
Case 02:
x_tilde[i] - x_tilde[i-1] > R == 0 ;
x_tilde[i] - x_tilde[i-1] > L ;
x_tilde[i] - x_tilde[i-1] > Gamma[i][0] // first element of Gamma[i]
'''
beta = L
beta_prime = L*L

'''
Case 03:
x_tilde[i] - x_tilde[i-1] > R != 0 ;
x_tilde[i] - x_tilde[i-1] > L ;
x_tilde[i] - x_tilde[i-1] > Gamma[i][0] // first element of Gamma[i]
'''
beta = L
beta_prime = L*L

'''
Case 04:
x_tilde[i] - x_tilde[i-1] > R != 0 ;
x_tilde[i] - x_tilde[i-1] > L ;
x_tilde[i] - x_tilde[i-1] <= Gamma[i][0] // first element of Gamma[i]
'''
beta = L
beta_prime = L*L

'''
Case 05:
x_tilde[i] - x_tilde[i-1] > R != 0 ;
x_tilde[i] - x_tilde[i-1] <= L ;
x_tilde[i] - x_tilde[i-1] > Gamma[i][0] // first element of Gamma[i]
'''


'''
Case 06:
x_tilde[i] - x_tilde[i-1] > R != 0 ;
x_tilde[i] - x_tilde[i-1] <= L ;
x_tilde[i] - x_tilde[i-1] <= Gamma[i][0] // first element of Gamma[i]
'''
for i in range(len(Omega[0])):
    if x[0] == Omega[0][i]:
        # print('Hello: ', i)
        beta1 = L - i
        # print('beta_01: ', beta1)

'''
Case 07:
x_tilde[i] - x_tilde[i-1] <= R ;
x_tilde[i] - x_tilde[i-1] <= L ;
'''
# x_2 = x_1 + (x_tilde[i] - x_tilde[i-1])

'''
Case 08:
x_tilde[i] - x_tilde[i-1] <= R ;
x_tilde[i] - x_tilde[i-1] > L ;
'''
beta = 1
beta_prime = L


Gamma = getGamma(R, L, x_tilde)
print('Gamma:')
print(Gamma)

def beta_01(x, i):

    beta_01 = 0
    difference = x_tilde[1] - x_tilde[0]

    print('L: ', L)
    print('R: ', R)

    # Case 01:
    if R == 0 and difference > R and difference <= L and difference > Gamma[1][0]:
        print('Hello, Case 01')
        beta_01 = L
        return beta_01

    # Case 02:
    if R == 0 and difference > R and difference > L and difference > Gamma[1][0]:
        print('Hello, Case 02')
        beta_01 = L
        return beta_01

    # Case 03:
    if R != 0 and difference > R and difference > L and difference > Gamma[1][0]:
        print('Hello, Case 03')
        beta_01 = L
        return beta_01

    # Case 04:
    if R != 0 and difference > R and difference > L and difference <= Gamma[1][0]:
        print('Hello, Case 04')
        beta_01 = L
        return beta_01

    # Case 05:
    if R != 0 and difference > R and difference <= L and difference > Gamma[1][0]:
        print('Hello, Case 03')
        beta_01 = L
        return beta_01

    # Case 06:
    if R != 0 and difference > R and difference <= L and difference <= Gamma[1][0]:
        print('Hello, Case 05')

    # Case 07:
    if R != 0 and difference <= R and difference <= L:
        print('Hello, Case 06')

    # Case 08:
    if R != 0 and difference <= R and difference > L:
        print('Hello, Case 07')

    # return beta_01

print('Beta_01: ', beta_01(x, 0))




'''
Resampling Distribution p(x), where x = (x_1,...,x_n)
p(x) := (1/Z)*h_1(x_1)Product{from i=2 to n}*h_i(x[i-1], x[i])
t = np.sort(np.random.randint(40, size=n+1))

def p(Z, i): # Z
    return (1/Z)*h_1(x[0])*h_i(i)

def p1(Z):
    return 1/Z

def sampling_distribution(size, L, R):
    p = 0
    obs_x = get_obs_x(size)
    x_tilde = get_x_tilde(obs_x)
    Omega = getOmega(L, x_tilde)
    Gamma = getGamma(R, L, x_tilde)

    # print("Obs_X: ", obs_x)
    print("X_tilde: ", x_tilde)
    print("Omega: ", Omega)
    print("Gamma: ", Gamma)
    # return p

print(sampling_distribution(10, 5, 10))
'''

'''
riter = list(range(1, 10))
random.shuffle(riter)
for i in riter:
    print('L: ', i)
    print(sampling_distribution(10, i, 3))
    print('\n')
'''

'''
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
'''

# Recursion Test:
def fibonacci(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    return fibonacci(n-2)+fibonacci(n-1)

# print('Fibonacci: ', fibonacci(20))

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


'''
# y = np.zeros(N)
y = np.ones(n)
m = len(obs_x)
plt.plot(x_tilde, y, 'o')
plt.xlim([0, m])

plt.ylim([-1, 1])
plt.axis([0, m, -1, 1])
plt.show()
'''

'''
print('Hello World!!')
print(Omega[:1,])
print(Omega[0:1,])
'''
'''
def isinteger(x):
    print('Integer')
    return np.equal(np.mod(x, 1), 0)

def isarray(vector):
    print('Array')
    if isinteger(vector):
        return False
    else:
        n = len(vector)
        if n > 1:
            return True

indicator = lambda x_i, Omega_i: 1 if x_i == Omega_i else 0
print(indicator(1,1))
'''

'''
# Iterate over Omega matrix columnwise
for i in Omega:
    print(i)
    for j in i:
        print(j)
'''

'''
np.arange(start, stop, step, dype=none)
#Return evenly spaced values within a given interval

np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None)
'''

'''
# Return evenly spaced numbers over a specified interval
# Testing Matplotlib in Scipy with linspace
N = 8
y = np.zeros(N)
x1 = np.linspace(0, 10, N, endpoint=True)
x2 = np.linspace(0, 10, N, endpoint=False)
plt.plot(x1, y, 'o')
plt.plot(x2, y + 0.5, 'o')
plt.ylim([-0.5, 1])
plt.show()
'''
