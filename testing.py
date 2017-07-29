
def getZeroPadding(Length):

    L = 0
    Zero = []
    L = Length
    # Create numpy.zeros(shape, dtype, order)
    zero_k = np.zeros(L, dtype=np.int)

    for i in range(L):
        Zero.append(zero_k)

    Zero = np.array(Zero)
    return Zero
# print(getZeroPadding(5))

# P(Z_2 | Z_1) <= Basis for Trnasition Matrices for Z = (Amount_S, Tj)
def getZdistBasis(tMatrix, where, Length):

    Sync = []
    NonSync = []
    ZeroPadding = []

    zDistBasis = []
    zDistBasis0 = []
    zDistBasis1 = []

    which = where
    m = which+1
    L = Length
    for i, Pi in enumerate(tMatrix[which]):

        nonSync_k = []
        sync_k = []

        for k, prob in enumerate(Pi):

            if syncStateMat[m][k] == 0:

                nonSync_k.append(prob)
                sync_k.append(0)

            else:

                nonSync_k.append(0)
                sync_k.append(prob)

        NonSync.append(nonSync_k)
        Sync.append(sync_k)

    ZeroPadding = getZeroPadding(L)

    Basis = np.concatenate((np.array(NonSync), np.array(Sync)), axis=1)
    zDistBasis0 = np.concatenate((Basis, ZeroPadding), axis=1)
    zDistBasis1 = np.concatenate((ZeroPadding, Basis), axis=1)

    zDistBasis.append(zDistBasis0)
    zDistBasis.append(zDistBasis1)
    zDistBasis = np.array(zDistBasis)

    return zDistBasis
# print(getZdistBasis(tDistMatrices, 5))

def getZdistMatrix(S, L, tDistMat):

    AmoutSync = S
    zDistBasis = getZdistBasis(tDistMat, S-2, L)
    ZeroPadding = getZeroPadding(L)
    len_zDistBasis = len(zDistBasis)

    zDist = []


    if AmoutSync != len_zDistBasis:

        add = []
        new = []
        get = []
        print('Hello There.')
        get = zDistBasis[S-(S-1)]
        # print(zDistBasis)

        get0 = []
        get1 = []
        get0 = zDistBasis[0]
        get1 = zDistBasis[1]
        newBasis0 = []
        newBasis1 = []

        for i in range(S-len_zDistBasis):
            # print(i)
            newBasis0 = np.concatenate((get0, ZeroPadding), axis=1)
            newBasis1 = np.concatenate((get1, ZeroPadding), axis=1)

            new = np.concatenate((ZeroPadding, get), axis=1)

            get = new
            get0 = newBasis0
            get1 = newBasis1
            # add.append(new)
            # add.append(np.concatenate((zDistBasis[i], ZeroPadding), axis=1))

        print(newBasis0)
        # print('add:')
        # add = np.array(add)
        # print(add)
        # add.append(np.array(new))
        # zDistBasis = np.array(add)
        # print('New Z Dist Basis: ')
        # print(zDistBasis)


            # for i in range(S-1):
            #
            #     add.append(np.concatenate((zDistBasis[i], ZeroPadding), axis=1))
            #
            # add.append(new)
            #
            # zDist = np.array(add)
            # zDistBasis = zDist

    else:

        zDist = zDistBasis

    return False
# print(getZdistMatrix(4, 5, tDistMatrices))
