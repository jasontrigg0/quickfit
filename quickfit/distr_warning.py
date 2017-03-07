import sklearn.base
import numpy as np

class DistrWarning(sklearn.base.BaseEstimator):
    """
    Part of sklearn pipeline
    on fit() it stores empty_pct, mean and sd information
    about all variables
    on transform() it prints warnings if any variable distributions
    look strange
    """
    def __init__(self):
        pass
    def fit(self, X, y=None):
        self.cnt = X.shape[0]
        self.means = X.mean()
        #use iqd instead of std for stability
        # self.stds = X.std()
        self.ile75 = X.quantile(0.75)
        self.ile25 = X.quantile(0.25)
        self.iqd = self.ile75 - self.ile25
        self.null_nan_frac = self.empty_frac(X)
        return self
    def transform(self, X):
        sample_cnt = X.shape[0]
        sample_means = X.mean()
        #use iqd instead of std for stability
        # sample_stds = X.std()
        sample_ile75 = X.quantile(0.75)
        sample_ile25 = X.quantile(0.25)
        sample_iqd = sample_ile75 - sample_ile25
        sample_null_nan_frac = self.empty_frac(X)

        def summary_stats(c):
            return [c,self.means[c],self.ile25[c],self.ile75[c],sample_means[c],sample_ile25[c],sample_ile75[c]]
        if self.cnt > 1000 and sample_cnt > 1000:
            #iqd changed too much:
            iqd_change = ((self.iqd != 0) & (sample_iqd == 0)) |   \
                         ((self.iqd == 0) & (sample_iqd != 0)) |   \
                         ((self.iqd != 0) & (sample_iqd != 0) & (abs(np.log(self.iqd) - np.log(sample_iqd)) > 0.7)) # sd_new / sd_old < 0.5 or > 2.0
            # print sample_iqd, np.log(sample_iqd)
            for c in X.columns[iqd_change]:
                sys.stderr.write("WARNING: test set column {:s} has large change in 25-75 percentile range. In training, mean,25%ile,75%ile = {:.4g},{:.4g},{:.4g}. Now: {:.4g},{:.4g},{:.4g}\n".format(*summary_stats(c)))

            #mean moved too much:
            # combo_iqd = (self.iqd ** 2 + sample_iqd ** 2) ** 0.5
            mean_diff = abs(self.means - sample_means)
            mean_change = (self.iqd > 0) & ((mean_diff / self.iqd) > 0.5)
            for c in X.columns[mean_change]:
                sys.stderr.write("WARNING: test set column {:s} has large change in mean. In training, mean,25%ile,75%ile = {:.4g},{:.4g},{:.4g}. Now: {:.4g},{:.4g},{:.4g}\n".format(*summary_stats(c)))



            #null frac changed too much:
            null_frac_change = ((self.null_nan_frac != 0) & (sample_null_nan_frac == 0)) |  \
                               ((self.null_nan_frac == 0) & (sample_null_nan_frac != 0)) |  \
                               (abs(self.null_nan_frac - sample_null_nan_frac) >= 0.4) | \
                               ((abs(self.null_nan_frac - sample_null_nan_frac) >= 0.05) & (abs(np.log(self.null_nan_frac) - np.log(sample_null_nan_frac)) > 1.5))
            for c in X.columns[null_frac_change]:
                sys.stderr.write("WARNING: test set column {:s} has large change in number empty. In training, empty & NaN fraction = {:.4g}. Now: {:.4g}\n".format(c,self.null_nan_frac[c],sample_null_nan_frac[c]))
        return X

    @classmethod
    def empty_frac(cls,X):
        return 1 - (X.count() / float(X.shape[0]))
