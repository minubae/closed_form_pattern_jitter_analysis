import numpy as np
from PatternJitter import *
from Data import *

givenT = getOmega(5, x_tilde)
print('Reference Train: ', Ref)
print('Given T: ')
# print(givenT)
upState = []
upStateIndex = []
index = 0

for j, Tj in enumerate(givenT):

    print(j, Tj)
    print('Reference[',j,']: ', Ref[j])
    index = np.where(Tj == Ref[j])[0]
    print('Index: ', index)
    print('Index Size: ', index.size, '\n')

    if index.size != 0:
        upStateIndex.append(np.asscalar(index))
    else:
        upStateIndex.append(False)

    values = np.in1d(Tj, Ref)
    # print('Where? : ', values, '\n')


print(upStateIndex)
