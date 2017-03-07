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
