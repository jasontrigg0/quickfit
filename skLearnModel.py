#!/usr/bin/env python
import pandas as pd
import numpy as np
import sklearn.pipeline
import random
import sys
import os
import utils
"""
SKLearnModel:
Basic
"""
def select_pseudorandom(df, subset, id_col=None):
    #TODO add id_column support (md5 hash the column)
    #TODO add warnings if validation, test, all run without sudo
    length = df.shape[0]
    random.seed(173)
    rand_list = [random.random() for i in range(length)]
    training = [i < 0.6 for i in rand_list]
    validation = [0.6 <= i < 0.8 for i in rand_list]
    test = [0.8 <= i for i in rand_list]
    val_and_test = [0.6 <= i for i in rand_list]
    if subset == "100":
        return df[training]
    elif subset == "010":
        return df[validation]
    elif subset == "001":
        return df[test]
    elif subset == "011":
        return df[val_and_test]
    elif subset == "111":
        return df
    else:
        raise Exception("unknown subset name {:s}. Try 100, 011, 111, etc".format(subset) + "\n")

def process_cut_csv(i,delim=","):
    if i:
        i = i.split(',')
        return list(process_cut_list(i))
    else:
        return None

def process_cut_list(l, delim=","):
    for i in l:
        if "-" in i:
            x,y = i.split('-')
            for r in range(int(x),int(y)+1):
                yield r
        elif utils.str_is_int(i):
            yield int(i)
        else:
            yield i

class QuickFitMixin(object):
    def readCL(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser = self.parent_args(parser)
        parser = self.add_subclass_args(parser)
        args = parser.parse_args()
        if args.full:
            args.fit_subset = "training"
            args.output_subset = "all"
            args.trim = True
        self.args = args

    def parent_args(self, parser):
        import argparse
        parser.add_argument("-f","--infile",help="name of data file", default=sys.stdin)
        parser.add_argument("-c","--keep_cols",help="feature columns to keep")
        parser.add_argument("-C","--drop_cols",help="feature_columns to drop")
        parser.add_argument("-t","--target",help="name of target variable or file", required=True)
        parser.add_argument("--full",action="store_true",help="more formal fitting, with trimming, leaving out validation and test sets, etc")
        parser.add_argument("--fit_subset", default="111", help="Data is split 60%%-20%%-20%% into training, validation and test sets. Select them using three 0-1 digits, one for each set. Example: '100' = training only. '001' = test only. '111' = all data")
        parser.add_argument("--pred_subset", default="111", help="Data is split 60%%-20%%-20%% into training, validation and test sets. Select them using three 0-1 digits, one for each set. Example: '100' = training only. '001' = test only. '111' = all data")
        parser.add_argument("--save_model_file")
        parser.add_argument("--load_model_file")
        parser.add_argument("--outfile", type=argparse.FileType('w'),help="File to write the prediction csv")
        parser.add_argument("--so",action="store_true",help="Write the prediction csv to /dev/stdout")
        parser.add_argument("--print_model", action="store_true")
        parser.add_argument("--score")
        parser.add_argument("--id_col", help="(optional) name of id column in the data")
        parser.add_argument("--fuzz", action="store_true", help="test model won't crash in production by replacing some data with random values")
        parser.add_argument("--trim",action="store_true", help="pretrim input variables to 5ile-95ile variables to protect against outliers")
        return parser

    def add_subclass_args(self, parser):
        #override in subclass
        pass

    def main(self):
        #run readCL if not already completed
        if not hasattr(self,"args"):
            self.readCL()
        # infile, keep_list, drop_list, target, fit_subset, output_subset, save_model_file, load_model_file, output_file, so, print_model, score, id_col, fuzz, no_trim = (self.args.infile, self.args.keep_cols, self.args.drop_cols, self.args.target, self.args.fit_subset, self.args.output_subset, self.args.save_model_file, self.args.load_model_file, self.args.output_file, self.args.so, self.args.print_model, self.args.score, self.args.id_col, self.args.fuzz, self.args.no_trim)
        if self.args.so:
            self.args.outfile = sys.stdout

        keep_cols = process_cut_csv(self.args.keep_cols)
        drop_cols = process_cut_csv(self.args.drop_cols)

        if self.args.trim:
            self.trim_frac = 0
        else:
            self.trim_frac = 0.05

        if sys.stdin.isatty() and self.args.infile == sys.stdin:
            sys.stderr.write("WARNING: pcsv using /dev/stdin as default input file (-f) but nothing seems to be piped in..." + "\n")

        df = pd.read_csv(self.args.infile, dtype="object")
        #feature_cols are used in fitting
        #output_cols are printed along with the prediction
        target = self.args.target
        if not target:
            feature_cols = df.columns
            output_cols = df.columns
        elif target in df.columns:
            feature_cols = [c for c in df.columns if c != target]
            output_cols = feature_cols
            if keep_cols:
                for c in keep_cols:
                    if c not in feature_cols:
                        raise Exception("ERROR: -c column '{c}' can't be found in input columns".format(**vars()))
                feature_cols = [c for c in feature_cols if c in keep_cols]
            if drop_cols:
                feature_cols = [c for c in feature_cols if c not in drop_cols]
            df_features = df[feature_cols]
            df_target = df[target]
        elif os.path.exists(target):
            feature_cols = df.columns
            output_cols = df.columns
            df_target = pd.read_csv(target, dtype="object").iloc[:,0]
        else:
            raise Exception("ERROR: unknown -t (--target) column or file '{target}'".format(**vars()))
        df_features = df[feature_cols]

        #either load or fit model
        if self.args.load_model_file:
            self.load_model(self.args.load_model_file)
        else:
            fit_features = select_pseudorandom(df_features, self.args.fit_subset)
            fit_target = select_pseudorandom(df_target, self.args.fit_subset)
            self.fit(fit_features, fit_target)

        #save model
        if self.args.save_model_file:
            self.save_model(self.args.save_model_file)

        #generate predictions
        df_output_rows = select_pseudorandom(df, self.args.pred_subset)
        pred_features = df_output_rows[feature_cols]
        if self.args.fuzz:
            pred_features = self.fuzz(pred_features)
        true_target = select_pseudorandom(df_target, self.args.pred_subset)

        true_target_np = true_target.astype(float).values #metrics functions need floats, not strings
        model_output = self.model_output(pred_features)

        if self.args.outfile:
            output_df = pd.concat([df_output_rows[output_cols], true_target, model_output],axis=1)
            output_df["error"] = model_output.iloc[:,0] - true_target_np
            output_df.to_csv(self.args.outfile, index=False)

        #print score
        pred_target_np = model_output.values
        import sklearn.metrics
        if len(df_target.unique()) == 2:
            sys.stderr.write("auroc: "+ str(sklearn.metrics.roc_auc_score(true_target_np, pred_target_np)) + "\n")
            sys.stderr.write("log_loss: " + str(sklearn.metrics.log_loss(true_target_np, pred_target_np)) + "\n")
        else:
            sys.stderr.write("r**2: " + str(sklearn.metrics.r2_score(true_target_np, pred_target_np)) + "\n")
            sys.stderr.write("sd_target: " + str(np.std(true_target_np)) + "\n")
            sys.stderr.write("sd_error: " + str(np.std(true_target_np - pred_target_np[:,0])) + "\n")

        if self.args.print_model:
            self.print_model()



class FittingMixin(object):
    @classmethod
    def readCL(cls):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--training_features")
        parser.add_argument("--training_target")
        parser.add_argument("--training_with_crossval", action="store_true")
        parser.add_argument("--test_features")
        parser.add_argument("--save_pred_file")
        parser.add_argument("--load_model_file")
        parser.add_argument("--save_model_file")
        parser.add_argument("--training_samplesize",help="number of training rows to sample", type=int)
        args = parser.parse_args()
        return args.training_features, args.training_target, args.training_with_crossval, args.test_features, args.save_pred_file, args.load_model_file, args.save_model_file, args.training_samplesize
    def main(self):
        training_features, training_target, training_crossval, test_features, save_pred_file, load_model_file, save_model_file, training_samplesize = self.readCL()

        if training_features and training_target:
            df_features = pd.read_csv(training_features)
            df_target = pd.read_csv(training_target).iloc[:,0] #scikit-learn wants 1d list for target var, not a 2d dataframe

            if training_samplesize:
                import random
                sample = random.sample(df_features.index, training_samplesize)
                df_features = df_features.ix[sample]
                df_target = df_target.ix[sample]

            self.train_model(df_features, df_target)
        if load_model_file:
            self.load_model(load_model_file)

        if test_features:
            df_features = pd.read_csv(test_features)
            pred = self.model_output(df_features)
        if save_pred_file:
            self.save_output(pred, save_pred_file)

        if save_model_file:
            self.save_model(save_model_file)



class SKLearnModel(object):
    def __init__(self, model=None, preprocessing_steps=None):
        if model:
            self.raw_mdl = model
        if preprocessing_steps:
            self.preprocessing_steps = preprocessing_steps
        else:
            self.preprocessing_steps = []
    def fit(self, df_features, df_target):
        self.N = len(df_target)
        if self.args.trim:
            trim_frac = 0.05
        else:
            trim_frac = 0

        preproc_steps = getattr(self, "preprocessing_steps", []) + [
            ('cleaner', DataCleaner()),
            ('distr', DistrWarning()),
        ]

        if self.no_scale:
            scaling_step = []
        else:
            scaling_step = [('scaler', sklearn.preprocessing.StandardScaler())]

        fitting_steps = [
            ('trimmer', Trimmer(trim_frac)),
            ('imputer', sklearn.preprocessing.Imputer(missing_values='NaN', strategy='mean', axis=0)),
        ] + scaling_step + [
            ('main_model',self.raw_mdl),
        ]

        pipeline_steps = preproc_steps + fitting_steps
        self.mdl = sklearn.pipeline.Pipeline(pipeline_steps)
        self.mdl.fit(df_features, df_target)
    def feature_names(self):
        return list(self.mdl.named_steps["cleaner"].get_feature_names())
    def feature_means(self):
        return list(self.mdl.named_steps["scaler"].mean_)
    def feature_stds(self):
        return list(self.mdl.named_steps["scaler"].scale_)
    def transform(self, X):
        return self.mdl.transform(X)
    def load_model(self, load_model_file):
        import cPickle
        self.mdl = cPickle.load(open(load_model_file,'rb'))
    def model_output(self, df_features):
        #classifier
        if hasattr(self.mdl, "predict_proba"):
            preds = self.mdl.predict_proba(df_features)

            #jtrigg@20150707 self.mdl.classes_ attribute
            #isn't working when self.mdl is a pipeline
            #and the last step of the pipeline is a gridsearchCV
            #so grab classes manually
            if isinstance(self.mdl, sklearn.pipeline.Pipeline) and isinstance(self.mdl.steps[-1][-1], sklearn.grid_search.GridSearchCV):
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
