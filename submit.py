class SubmissionPipeline:
    
    def __init__(self, train, test, blend, prediction_columns, add_public_best):
        
        self.train = train
        self.test = test
        self.blend = blend
        self.prediction_columns = prediction_columns
        self.add_public_best = add_public_best
        
    def weighted_average(self):
        
        self.train['FinalPredictions'] = (0.77 * self.train['LGBPredictions']) +\
                                         (0.04 * self.train['CBPredictions']) +\
                                         (0.06 * self.train['XGBPredictions']) +\
                                         (0.0 * self.train['RFPredictions']) +\
                                         (0.0 * self.train['RRPredictions']) +\
                                         (0.0 * self.train['SVMPredictions']) +\
                                         (0.12 * self.train['TMLPPredictions']) +\
                                         (0.01 * self.train['RMLPPredictions'])
        
        self.test['FinalPredictions'] = (0.77 * self.test['LGBPredictions']) +\
                                        (0.04 * self.test['CBPredictions']) +\
                                        (0.06 * self.test['XGBPredictions']) +\
                                        (0.0 * self.test['RFPredictions']) +\
                                        (0.0 * self.test['RRPredictions']) +\
                                        (0.0 * self.test['SVMPredictions']) +\
                                        (0.12 * self.test['TMLPPredictions']) +\
                                        (0.01 * self.test['RMLPPredictions'])
        
    def geometric_average(self):
        
        self.train['FinalPredictions'] = gmean(self.train[self.prediction_columns], axis=1)
        self.test['FinalPredictions'] = gmean(self.test[self.prediction_columns], axis=1)
        
    def weighted_average_public(self):
        
        public_best = pd.read_csv('submission2.csv')
        self.test['FinalPredictions'] = (self.test['FinalPredictions'] * 0.02) + (public_best['target'] * 0.98) 
        
    def transform(self):
        
        if self.blend == 'weighted_average':
            self.weighted_average()
        elif self.blend == 'geometric_average':
            self.geometric_average()
            
        for prediction_column in prediction_columns:
            oof_score = mean_squared_error(self.train[target], df_train_processed[prediction_column], squared=False)
            print(f'{prediction_column.split("Predictions")[0]} OOF RMSE: {oof_score:.6}')
        final_oof_score = mean_squared_error(self.train[target], df_train_processed['FinalPredictions'], squared=False)
        print(f'{"-" * 30}\nFinal OOF RMSE: {final_oof_score:.6}\n{"-" * 30}')
                
        if self.add_public_best:
            self.weighted_average_public()
                
        return self.train[['id'] + self.prediction_columns + ['FinalPredictions']].copy(deep=True), self.test[['id'] + self.prediction_columns + ['FinalPredictions']].copy(deep=True)
            


submission_pipeline = SubmissionPipeline(train=df_train_processed, test=df_test_processed,
                                         blend='weighted_average', prediction_columns=prediction_columns, add_public_best=True)   
df_train_submission, df_test_submission = submission_pipeline.transform()

df_test_processed['target'] = df_test_submission['FinalPredictions']
df_test_processed[['id', 'target']].to_csv('submission.csv', index=False)
df_test_processed[['id', 'target']].describe()