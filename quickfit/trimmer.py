import sklearn.base

class Trimmer(sklearn.base.BaseEstimator):
    """
    cap variables at x%ile and (100-x)%ile for some x
    """
    def __init__(self, pctile=0.05):
        if pctile > 1: raise
        self.pctile = pctile
    def fit(self, X, y=None):
        self.mins = X.quantile(self.pctile)
        self.maxs = X.quantile(1 - self.pctile)
        return self
    def transform(self, X):
        for i,c in enumerate(X.columns):
            X.loc[X[c] < self.mins[i],c] = self.mins[i]
            X.loc[X[c] > self.maxs[i],c] = self.maxs[i]
        return X
