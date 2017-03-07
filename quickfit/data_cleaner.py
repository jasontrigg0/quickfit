import pandas as pd

def float_or_nan(var):
    try:
        f = float(var)
        return True
    except:
        return False


class DataCleaner(object):
    """
    Part of sklearn pipeline
    - removes non-numeric columns
    - removes empty columns
    - reorders columns in alphabetical order (because later steps may only use the underlying numpy array)
    - replaces non-numeric values in numeric columns with NaN
    - convert remaining columns to numeric
    """
    def __init__(self):
        pass
    def fit(self, X,y=None):
        non_numeric_cols = X.applymap(lambda x: not float_or_nan(x)).any(axis=0)
        self.non_numeric_cols = sorted(X.columns[non_numeric_cols])
        if any(self.non_numeric_cols):
            sys.stderr.write("WARNING: removing non-numeric columns " + str(self.non_numeric_cols) + "\n")

        empty_cols = (pd.isnull(X).all())
        self.empty_cols = sorted(X.columns[empty_cols])
        if any(self.empty_cols):
            sys.stderr.write("WARNING: removing empty columns " + str(self.empty_cols) + "\n")

        self.feature_names = sorted(X.columns[~non_numeric_cols & ~empty_cols])
        if len(self.feature_names) == 0:
            raise Exception("ERROR: no valid columns to train on!")
        return self
    def transform(self, X):
        #keep only columns from self.feature_names
        to_drop = set(X.columns).difference(set(self.feature_names))
        if any(to_drop):
            sys.stderr.write("WARNING: dropping extra columns " + str(to_drop) + "\n")
        X = X.drop(to_drop, axis=1)

        #all remaining columns should be numeric:
        #replace non-numeric values with np.nan
        X = X.applymap(lambda x: x if float_or_nan(x) else np.nan)

        #rearrange columns in alphabetical order, in case order changes
        X = X[sorted(X.columns)]

        if (list(X.columns) != self.feature_names):
            missing_cols = str(list(set(self.feature_names).difference(set(X.columns))))
            raise Exception("ERROR: missing columns {missing_cols}".format(**vars()))
        # assert(list(X.columns) == self.feature_names) #check correct columns are in place
        return X.astype("float")
    def get_params(self, deep=False):
        return {}
    def get_feature_names(self):
        return self.feature_names
