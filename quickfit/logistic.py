#!/usr/bin/env python
import sys
from quickfit.precleaned_model import PrecleanedModel
import sklearn.linear_model

class LogisticReg(PrecleanedModel):
    def __init__(self):
        #scoring "log_loss" needed here
        #(because default is "accuracy_score" and logistic doesn't make a class prediction?)
        #TODO(jtrigg): SHOULD REFIT BE TRUE OR FALSE? I thought true, but false is giving more reasonable results right now
        self.raw_mdl = sklearn.linear_model.LogisticRegressionCV(penalty='l1', solver="liblinear", scoring="neg_log_loss",refit=False)

        #call super after the raw_mdl is set
        super(LogisticReg, self).__init__()
    def print_model(self):
        #get feature names from dataCleaner object
        feature_names = ["(intercept)"] + list(self.mdl.named_steps["cleaner"].get_feature_names())
        coeffs = list(self.mdl.steps[-1][1].intercept_) + list((self.mdl.steps[-1][1].coef_)[0])
        sys.stdout.write("Regularization C=" + str(self.raw_mdl.C_[0]) + '\n')
        sys.stdout.write("Normalized coefficients:" + '\n')
        sys.stdout.write(",".join([str(i) for i in feature_names]) + '\n')
        sys.stdout.write(",".join([str(i) for i in coeffs]) + '\n')
    def add_subclass_args(self, parser):
        # parser.add_argument("-a","--alpha",type=float)
        parser.add_argument("--no_scale",action="store_true")
        # parser.add_argument("--no_intercept",action="store_true")
        return parser
