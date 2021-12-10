def task(x,y):
    import numpy as np
    # return x[0].ravel().shape,y.ravel().shape
    # return x[0].ravel(),y.ravel(),x[0].ravel().shape,y.ravel().shape
    return np.corrcoef(x[0].ravel(),y.ravel())[0,1]
    # print(x,y)

    # return np.corrcoef(x.ravel(),y.ravel())[0,1]
