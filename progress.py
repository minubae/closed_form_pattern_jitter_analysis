import numpy as np

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

sqs_spike_time_x = np.array(x)

# Jitter Window
Omega = []
n = len(sqs_spike_time_x)
for i in range(n):
    for j in range(1, L+1):
        Omega.append(sqs_spike_time_x[i] - np.floor(L/2) + j)


print(obs_x)
print(sqs_spike_time_x)
print(Omega)
