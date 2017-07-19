import numpy as np

# x = np.random.choice(np.arange(1, 7), p=[0.1, 0.05, 0.05, 0.2, 0.4, 0.2])
# print('x: ', x)

tDist = [0.2, 0.1, 0.15, 0.15, 0.25, 0.15]
def roll(distribution):
    print('Transition Dist.:', distribution)
    randRoll = np.random.random() # in [0,1)
    print('Random Roll:', randRoll)
    sum = 0
    result = 1
    for k in distribution:
        print('Dist:', k)
        sum += k
        if randRoll <= sum:
            print('Sum:', sum)
            return result
        result+=1

print('X:', roll(tDist))
