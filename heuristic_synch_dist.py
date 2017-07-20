import numpy as np
import matplotlib.pyplot as plt


T = [2,3,4,5,6,3,4,3,5,6,3,4,3,4,5,6,3,]
plt.hist(T, bins='auto')
# plt.title("Histogram with 'auto' bins")
print('T: ', T)
plt.show()


'''
x = np.random.normal(size = 1000)
plt.hist(x, normed=True, bins=30)
plt.ylabel('Probability');
plt.show()
'''
