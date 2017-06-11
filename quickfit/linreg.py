#!/usr/bin/env python
from precleaned_model import PrecleanedModel
import pandas as pd
import sys
import sklearn.linear_model
from utils import pretty_print_df #TODO(jtrigg) include these function in this package or in jtutils
import argparse
from pcsv.any2csv import df2pretty

class LinReg(PrecleanedModel):
    def __init__(self):
        self.raw_mdl = sklearn.linear_model.LassoCV()

        #call super after raw_mdl is set
        super(LinReg, self).__init__()
    def model_desc(self):
        #get feature names from dataCleaner object
        feature_names = ["(Intercept)"] + self.feature_names()
        normed_coeffs = self.coeffs()
        normed_intercept = self.intercept()

        if not self.no_scale:
            means = self.feature_means()
            sds = self.feature_stds()
            raw_coeffs = [coef / sd for (coef,sd,mean) in zip(normed_coeffs, sds, means)]
            raw_intercept = normed_intercept[0] - sum([coef * mean for (coef,mean) in zip(raw_coeffs, means)])
            feature_coeffs = [raw_intercept] + raw_coeffs
        else:
            raw_coeffs = normed_coeffs
            raw_intercept = normed_intercept
            feature_coeffs = raw_intercept + raw_coeffs
        return feature_names, feature_coeffs
    def model_df(self):
        feature_names, feature_coeffs = self.model_desc()
        return pd.DataFrame([feature_coeffs], columns = feature_names)
    def model_dict(self):
        feature_names, feature_coeffs = self.model_desc()
        return dict(zip(feature_names, feature_coeffs))
    def coeffs(self):
        return list((self.mdl.steps[-1][1].coef_))
    def intercept(self):
        return list([self.mdl.steps[-1][1].intercept_])
    def print_model(self):
        sys.stderr.write("N: " + str(self.N) + "\n")
        if self.args.alpha is None: #alpha determined by cv
            sys.stderr.write("alpha: " + str(self.mdl.steps[-1][1].alpha_) + "\n")
            each_fold_mse = self.mdl.steps[-1][1].mse_path_[-1]
            overall_mse = sum(each_fold_mse) / float(len(each_fold_mse))
            rmse = overall_mse ** 0.5
            sys.stderr.write("cross-validation training error: " + str(rmse) + "\n")
        if not self.no_scale:
            sys.stderr.write('\n' + "-----" + '\n')
            sys.stderr.write("Feature means:" + '\n')
            means = pd.DataFrame([self.feature_means()], columns = self.feature_names())
            sys.stderr.write(df2pretty(means))
            sys.stderr.write( '\n' + "-----" + '\n')
            sys.stderr.write("Feature sds:" + '\n')
            sds = pd.DataFrame([self.feature_stds()], columns = self.feature_names())
            sys.stderr.write(df2pretty(sds))
            sys.stderr.write( '\n' + "-----" + '\n')
            sys.stderr.write("Normalized coefficients:" + '\n')
            coeffs = pd.DataFrame([self.coeffs()], columns = self.feature_names())
            sys.stderr.write(df2pretty(coeffs))
        sys.stderr.write('\n' + "-----"  + '\n')
        sys.stderr.write("Raw coefficients:" + '\n')
        sys.stderr.write(df2pretty(self.model_df()))
        sys.stderr.write('\n' + str(self.model_dict()) + "\n")
    def add_subclass_args(self, parser):
        parser.add_argument("-a","--alpha",type=float)
        parser.add_argument("--no_scale",action="store_true")
        parser.add_argument("--no_intercept",action="store_true")
        return parser
