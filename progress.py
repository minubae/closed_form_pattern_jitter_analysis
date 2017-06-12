import numpy as np

# Observed Spike Train
# x = np.random.uniform(0,1,(6,6))
# x = np.random.uniform(0,1,(6,6))
x = np.random.randint(2, size=10)
size = len(x)
L = 2

for i in range(0, size, L):
    print(i)
    #print(x[i])

print(x)
