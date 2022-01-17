class LinearModels:
    
    def __init__(self, predictors, target, model, model_parameters):
        
        self.predictors = predictors
        self.target = target
               
        self.model = model
        self.model_parameters = model_parameters       
                
    def _train_and_predict_ridge_regression(self, X_train, y_train, X_test):
                
        X = pd.concat([X_train[continuous_features], X_test[continuous_features]], ignore_index=True, axis=0)
        embedder = RandomTreesEmbedding(n_estimators=800,
                                        max_depth=8,
                                        min_samples_split=100,
                                        n_jobs=-1,
                                        random_state=0, 
                                        verbose=False)
        embedder.fit(X)        
        del X
        X_test = embedder.transform(X_test.loc[:, continuous_features]).astype(np.uint8)
                        
        oof_predictions = np.zeros(X_train.shape[0])
        test_predictions = np.zeros(X_test.shape[0])
        
        for fold in sorted(X_train['fold'].unique()):
            
            trn_idx, val_idx = X_train.loc[X_train['fold'] != fold].index, X_train.loc[X_train['fold'] == fold].index

            X_trn = embedder.transform(X_train.loc[trn_idx, continuous_features]).astype(np.uint8)
            y_trn = y_train.loc[trn_idx]
            X_val = embedder.transform(X_train.loc[val_idx, continuous_features]).astype(np.uint8)
            y_val = y_train.loc[val_idx]
            
            model = Ridge(**self.model_parameters)
            model.fit(X_trn, y_trn)
            
            val_predictions = model.predict(X_val)
            oof_predictions[val_idx] += val_predictions
            fold_test_predictions = model.predict(X_test)
            test_predictions += (fold_test_predictions / X_train['fold'].nunique())
                        
            fold_score = mean_squared_error(y_val, val_predictions, squared=False)
            print(f'RR Fold {int(fold)} - X_trn: {X_trn.shape} X_val: {X_val.shape} - Score: {fold_score:.6}')
            
        df_train_processed['RRPredictions'] = oof_predictions
        df_test_processed['RRPredictions'] = test_predictions
        oof_score = mean_squared_error(y_train, df_train_processed['RRPredictions'], squared=False)
        print(f'{"-" * 30}\nRR OOF RMSE: {oof_score:.6}\n{"-" * 30}')
        
        self._plot_predictions(df_train_processed[target], df_train_processed['RRPredictions'], df_test_processed['RRPredictions'])
            
    def _train_and_predict_svm(self, X_train, y_train, X_test):
        
        oof_predictions = np.zeros(X_train.shape[0])
        test_predictions = np.zeros(X_test.shape[0])
        
        for fold in sorted(X_train['fold'].unique()):
            
            trn_idx, val_idx = X_train.loc[X_train['fold'] != fold].index, X_train.loc[X_train['fold'] == fold].index

            X_trn = X_train.loc[trn_idx, self.predictors]
            y_trn = y_train.loc[trn_idx]
            X_val = X_train.loc[val_idx, self.predictors]
            y_val = y_train.loc[val_idx]

            import cuml
            model = cuml.SVR(**self.model_parameters)
            model.fit(X_trn, y_trn)
            
            val_predictions = model.predict(X_val)
            oof_predictions[val_idx] += val_predictions
            fold_test_predictions = model.predict(X_test)
            test_predictions += (fold_test_predictions / X_train['fold'].nunique())
            
            fold_score = mean_squared_error(y_val, val_predictions, squared=False)
            print(f'SVM Fold {int(fold)} - X_trn: {X_trn.shape} X_val: {X_val.shape} - Score: {fold_score:.6}')
            
        df_train_processed['SVMPredictions'] = oof_predictions
        df_test_processed['SVMPredictions'] = test_predictions
        oof_score = mean_squared_error(y_train, df_train_processed['SVMPredictions'], squared=False)
        print(f'{"-" * 30}\nSVM OOF RMSE: {oof_score:.6}\n{"-" * 30}')
        
        self._plot_predictions(df_train_processed[target], df_train_processed['SVMPredictions'], df_test_processed['SVMPredictions'])
        
    def _plot_predictions(self, train_labels, train_predictions, test_predictions):
        
        fig, axes = plt.subplots(ncols=2, figsize=(25, 6))                                            
        sns.scatterplot(train_labels, train_predictions, ax=axes[0])
        sns.distplot(train_predictions, label='Train Predictions', ax=axes[1])
        sns.distplot(test_predictions, label='Test Predictions', ax=axes[1])

        axes[0].set_xlabel(f'Train Labels', size=18)
        axes[0].set_ylabel(f'Train Predictions', size=18)
        axes[1].set_xlabel('')
        axes[1].legend(prop={'size': 18})
        for i in range(2):
            axes[i].tick_params(axis='x', labelsize=15)
            axes[i].tick_params(axis='y', labelsize=15)
        axes[0].set_title(f'Train Labels vs Train Predictions', size=20, pad=20)
        axes[1].set_title(f'Predictions Distributions', size=20, pad=20)
            
        plt.show() 
            
    def run(self, X_train, y_train, X_test):
        
        if self.model == 'Ridge':
            self._train_and_predict_ridge_regression(X_train, y_train, X_test)
        elif self.model == 'SVM':
            self._train_and_predict_svm(X_train, y_train, X_test)