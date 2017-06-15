import numpy as np
import itertools as itt

# Observed Spike Train
# x = np.random.uniform(0,1,(6,6))
# Generating a binary random spike train with size = n
obs_x = np.random.randint(2, size=20)
size = len(obs_x)
L = 4

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
Omega = []
n = len(x_tilde)
for i in range(n):
    Omega[i].append(x_tilde[i] - np.floor(L/2) + j)

# for j in range(1, L+1):
#     Omega.append(x_tilde[i] - np.floor(L/2) + j)

my_array = np.empty([3,3])
for i,j in itt.product(range(3), range(3)):
    my_array[i,j] = f(i,j)

#print('Hello World!!')
print("Observed_X: ", obs_x)
print("x_tilde: ", x_tilde)
print("My Array: ", my_array)
#print(Omega)
