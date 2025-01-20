import pandas as pd
import random
import sys
import numpy as np
from jtutils import str_is_int
import os
"""
QuickFitMixin adds support for quick use at the command line
Allows easy model load/save and breaking into train/test sets
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
        elif str_is_int(i):
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
        parser.add_argument("-w","--weights",help="name of weights variable")
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
        return parser

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
            fit_weights = None

            if self.args.weights:
                weights = self.args.weights
                df_weights = df[weights]
                fit_weights = select_pseudorandom(df_weights, self.args.fit_subset)
                
            self.fit(fit_features, fit_target, sample_weight=fit_weights)

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
        model_output = self.transform(pred_features)

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
