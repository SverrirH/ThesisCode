


class neutral_standardizer():
    def __init__(self): pass
    def fit(self, X, Y=None, sample_weight=None): return self
    def transform(self, x): return x
    
class persistence():
    def __init__(self): pass
    def fit(self, X, Y, sample_weight=None): pass
    def predict(self, x): 
        y = x.copy()
        y[:, :] = y[:, [0], :]
        return y
