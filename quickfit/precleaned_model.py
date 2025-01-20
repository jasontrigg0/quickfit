#!/usr/bin/env python
import pandas as pd
import sklearn.pipeline
import sklearn.model_selection
import sklearn.impute
import random
import numpy as np
from .data_cleaner import DataCleaner
from .distr_warning import DistrWarning
from .trimmer import Trimmer
from collections import defaultdict
"""
A few quick preprocessing steps to run before passing raw data into a standard model (eg Lasso regression)
For example: rescale variables to std=1, remove non-numeric values (and impute their true value), trim outliers
"""

class PrecleanedModel(object):
    def __init__(self, model=None, preprocessing_steps=None):
        if model:
            self.raw_mdl = model
        if preprocessing_steps:
            self.preprocessing_steps = preprocessing_steps
        else:
            self.preprocessing_steps = []
        self.model_setup()
    def model_setup(self):
        if hasattr(self, "args") and self.args.trim:
            trim_frac = 0.05
        else:
            trim_frac = 0

        preproc_steps = getattr(self, "preprocessing_steps", []) + [
            ('cleaner', DataCleaner()),
            ('distr', DistrWarning()),
        ]

        if hasattr(self, "no_scale") and self.no_scale:
            scaling_step = []
        else:
            scaling_step = [('scaler', sklearn.preprocessing.StandardScaler())]

        fitting_steps = [
            ('trimmer', Trimmer(trim_frac)),
            ('imputer', sklearn.impute.SimpleImputer(missing_values=np.nan,strategy='mean')),
        ] + scaling_step + [
            ('main_model',self.raw_mdl),
        ]

        pipeline_steps = preproc_steps + fitting_steps
        self.mdl = sklearn.pipeline.Pipeline(pipeline_steps)
    def fit(self, df_features, df_target, sample_weight=None):
        self.N = len(df_target)
        self.mdl.fit(df_features, df_target, **{'main_model__sample_weight': sample_weight})
    def feature_names(self):
        return list(self.mdl.named_steps["cleaner"].get_feature_names())
    def feature_means(self):
        return list(self.mdl.named_steps["scaler"].mean_)
    def feature_stds(self):
        return list(self.mdl.named_steps["scaler"].scale_)
    def load_model(self, load_model_file):
        import cPickle
        self.mdl = cPickle.load(open(load_model_file,'rb'))
    def transform(self, df_features):
        #classifier
        if hasattr(self.mdl, "predict_proba"):
            preds = self.mdl.predict_proba(df_features)

            #jtrigg@20150707 self.mdl.classes_ attribute
            #isn't working when self.mdl is a pipeline
            #and the last step of the pipeline is a gridsearchCV
            #so grab classes manually
            if isinstance(self.mdl, sklearn.pipeline.Pipeline) and isinstance(self.mdl.steps[-1][-1], sklearn.model_selection.GridSearchCV):
                classes = self.mdl.steps[-1][-1].best_estimator_.classes_
            else:
                classes = self.mdl.classes_

            class1_index = list(classes).index("1")
            #print out only the class="1" probability
            preds = pd.DataFrame(preds[:,class1_index], columns=["pred"])
        else:
            preds = self.mdl.predict(df_features)
        if hasattr(preds,"values"):
            return pd.DataFrame(preds.values,columns=["pred"])
        else:
            return pd.DataFrame(preds, columns=["pred"])
    def fit_transform(self, X, y, sample_weight=None):
        self.fit(X,y, **{'main_model__sample_weight': sample_weight})
        return self.transform(X)
    @classmethod
    def save_output(cls, preds, save_pred_file):
        preds.to_csv(save_pred_file, index=False)
    def save_model(self, save_model_file):
        import cPickle
        cPickle.dump(self.mdl, open(save_model_file,'wb'))
    def set_preprocessing(self, preprocessing_steps):
        self.preprocessing_steps = preprocessing_steps
    def model_desc(self):
        pass
    @classmethod
    def fuzz(cls, df_features):
        def _rand_val():
            r = random.random()
            if r < 0.3:
                return "".join([chr(random.randint(0,255)) for i in range(10)])
            elif r < 0.9:
                return str(random.gauss(0,1))
            else:
                return ""
        return df_features.applymap(lambda x: _rand_val())
