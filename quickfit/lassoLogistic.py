#!/usr/bin/env python
import skLearnModel
import pandas as pd
import sys
import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model


class LassoLogistic(skLearnModel.SKLearnModel, skLearnModel.QuickFitMixin):
    def __init__(self, C=1):
        self.raw_mdl = sklearn.linear_model.LogisticRegression(C=C, penalty='l1')
    def model_desc(self):
        #get feature names from dataCleaner object
        feature_names = ["(intercept)"] + list(self.mdl.named_steps["cleaner"].get_feature_names())
        coeffs = list(self.mdl.steps[-1][1].intercept_) + list((self.mdl.steps[-1][1].coef_)[0])
        return pd.DataFrame([coeffs], columns = feature_names)
    #','.join([str(i) for i in feature_names]) + '\n' +\
    #        ",".join([str(i) for i in coeffs]) + '\n'

    def print_model(self):
        self.model_desc().to_csv(sys.stdout, index=False)
        # sys.stdout.write(self.model_desc())
    
        

if __name__ == "__main__":
    a = LassoLogistic(0.4)
    a.main()
    a.print_model()













# import argparse
# import cPickle
# import pandas as pd
# import numpy as np
# import sklearn
# import sklearn.pipeline
# import sklearn.linear_model
# import sklearn.grid_search

# def readCL():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--training_features")
#     parser.add_argument("--training_target")
#     parser.add_argument("--training_with_crossval", action="store_true")
#     parser.add_argument("--test_features")
#     parser.add_argument("--save_pred_file")
#     parser.add_argument("--load_model_file")
#     parser.add_argument("--save_model_file")
#     args = parser.parse_args()
#     return args.training_features, args.training_target, args.training_with_crossval, args.test_features, args.save_pred_file, args.load_model_file, args.save_model_file

# def train_model(training_features, training_target, crossval=True):
#     df_features = pd.read_csv(training_features)
#     df_target = pd.read_csv(training_target).iloc[:,0] #scikit-learn wants 1d list for target var, not a 2d dataframe

#     mdl = sklearn.pipeline.Pipeline([
#         ('scale', sklearn.preprocessing.StandardScaler()),
#         ('logistic', sklearn.linear_model.LogisticRegression(C=1e6)),
#     ])

#     if crossval:
#         #to specify both which step and which parameter in a pipeline grid_search use '{step_name}__{parameter_name}'
#         param_grid = {"logistic__C":np.logspace(-4,4, num=9)} #10**-4, 10**-3, ... 10**3, 10**4

#         #scoring parameter choices here:
#         #http://scikit-learn.org/stable/modules/model_evaluation.html
#         #'log_loss' is the penalty used in logistic_regression
#         grid = sklearn.grid_search.GridSearchCV(mdl, param_grid=param_grid, scoring='log_loss')
#         grid.fit(df_features, df_target)

#         mdl = grid.best_estimator_

#         #LogisticRegressionCV only available on newer versions of scikit-learn -- tencent probably doesn't have it
#         # mdl = sklearn.linear_model.LogisticRegressionCV()
#     else:
#         mdl.fit(df_features, df_target)
#         print mdl.coef_
#         print dir(mdl)
#         print mdl.score(df_features, df_target)
#     return mdl
    
# def load_model(load_model_file):
#     mdl = cPickle.load(open(load_model_file,'rb'))
#     return mdl
    
# def pred_model(mdl, test_features):
#     df_features = pd.read_csv(test_features)
#     preds = mdl.predict_proba(df_features)
#     class1_index = list(mdl.classes_).index(1)
#     preds = pd.DataFrame(preds[:,class1_index], columns=["pred"])
#     return preds
    
# def save_preds(preds, save_pred_file):
#     preds.to_csv(save_pred_file, index=False)

# def save_model(mdl, save_model_file):
#     cPickle.dump(mdl, open(save_model_file,'wb'))
    
# if __name__ == "__main__":
#     training_features, training_target, training_crossval, test_features, save_pred_file, load_model_file, save_model_file = readCL()
#     if training_features and training_target:
#         mdl = train_model(training_features, training_target, training_crossval)
#     if load_model_file:
#         mdl = load_model(load_model_file)

#     if test_features:
#         pred = pred_model(mdl, test_features)
#     if save_pred_file:
#         save_preds(pred, save_pred_file)

#     if save_model_file:
#         save_model(mdl, save_model_file)
