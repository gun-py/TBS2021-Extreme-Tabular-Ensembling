class TreeModels:
    def __init__(self, predictors, target, model, model_parameters, boosting_rounds, early_stopping_rounds, seeds):
        
        self.predictors = predictors
        self.target = target
               
        self.model = model
        self.model_parameters = model_parameters
        self.boosting_rounds = boosting_rounds
        self.early_stopping_rounds = early_stopping_rounds
        self.seeds = seeds
                
    def _train_and_predict_lgb(self, X_train, y_train, X_test):
        
        seed_avg_oof_predictions = np.zeros(X_train.shape[0])
        seed_avg_test_predictions = np.zeros(X_test.shape[0])        
        seed_avg_importance = pd.DataFrame(data=np.zeros(len(self.predictors)), index=self.predictors, columns=['Importance'])
        
        for seed in self.seeds:
            print(f'{"-" * 30}\nRunning LightGBM model with seed: {seed}\n{"-" * 30}\n')
            self.model_parameters['seed'] = seed
            self.model_parameters['feature_fraction_seed'] = seed
            self.model_parameters['bagging_seed'] = seed
            self.model_parameters['drop_seed'] = seed
            self.model_parameters['data_random_seed'] = seed
                
            for fold in sorted(X_train['fold'].unique()):

                trn_idx, val_idx = X_train.loc[X_train['fold'] != fold].index, X_train.loc[X_train['fold'] == fold].index
                trn = lgb.Dataset(X_train.loc[trn_idx, self.predictors], label=y_train.loc[trn_idx])
                val = lgb.Dataset(X_train.loc[val_idx, self.predictors], label=y_train.loc[val_idx])

                model = lgb.train(params=self.model_parameters,
                                  train_set=trn,
                                  valid_sets=[trn, val],
                                  num_boost_round=self.boosting_rounds,
                                  early_stopping_rounds=self.early_stopping_rounds,
                                  verbose_eval=500)            

                val_predictions = model.predict(X_train.loc[val_idx, self.predictors])
                seed_avg_oof_predictions[val_idx] += (val_predictions / len(self.seeds))
                test_predictions = model.predict(X_test[self.predictors])
                seed_avg_test_predictions += (test_predictions / X_train['fold'].nunique() / len(self.seeds))
                seed_avg_importance['Importance'] += (model.feature_importance(importance_type='gain') / X_train['fold'].nunique() / len(self.seeds))

                fold_score = mean_squared_error(y_train.loc[val_idx], val_predictions, squared=False)
                print(f'\nLGB Fold {int(fold)} - X_trn: {X_train.loc[trn_idx, self.predictors].shape} X_val: {X_train.loc[val_idx, self.predictors].shape} - Score: {fold_score:.6} - Seed: {seed}\n')
            
        df_train_processed['LGBPredictions'] = seed_avg_oof_predictions
        df_test_processed['LGBPredictions'] = seed_avg_test_predictions
        oof_score = mean_squared_error(y_train, df_train_processed['LGBPredictions'], squared=False)
        print(f'{"-" * 30}\nLGB OOF RMSE: {oof_score:.6} ({len(self.seeds)} Seed Average)\n{"-" * 30}')
                
        self._plot_importance(seed_avg_importance)
        self._plot_predictions(df_train_processed[target], df_train_processed['LGBPredictions'], df_test_processed['LGBPredictions'])
        
    def _train_and_predict_cb(self, X_train, y_train, X_test):
        
        seed_avg_oof_predictions = np.zeros(X_train.shape[0])
        seed_avg_test_predictions = np.zeros(X_test.shape[0])        
        seed_avg_importance = pd.DataFrame(data=np.zeros(len(self.predictors)), index=self.predictors, columns=['Importance'])
            
        for seed in self.seeds:
            print(f'{"-" * 30}\nRunning CatBoost model with seed: {seed}\n{"-" * 30}\n')
            self.model_parameters['random_seed'] = seed
            
            for fold in sorted(X_train['fold'].unique()):

                trn_idx, val_idx = X_train.loc[X_train['fold'] != fold].index, X_train.loc[X_train['fold'] == fold].index
                trn = cb.Pool(X_train.loc[trn_idx, self.predictors], label=y_train.loc[trn_idx])
                val = cb.Pool(X_train.loc[val_idx, self.predictors], label=y_train.loc[val_idx])

                model = cb.CatBoostRegressor(**self.model_parameters)
                model.fit(X=trn, eval_set=val)

                val_predictions = model.predict(val)
                seed_avg_oof_predictions[val_idx] += (val_predictions / len(self.seeds))
                test_predictions = model.predict(cb.Pool(X_test[self.predictors]))
                seed_avg_test_predictions += (test_predictions / X_train['fold'].nunique() / len(self.seeds))
                seed_avg_importance['Importance'] += (model.get_feature_importance() / X_train['fold'].nunique() / len(self.seeds))

                fold_score = mean_squared_error(df_train_processed.loc[val_idx, self.target], val_predictions, squared=False)
                print(f'\nCB Fold {int(fold)} - X_trn: {X_train.loc[trn_idx, self.predictors].shape} X_val: {X_train.loc[val_idx, self.predictors].shape} - Score: {fold_score:.6} - Seed: {seed}\n')
            
        df_train_processed['CBPredictions'] = seed_avg_oof_predictions
        df_test_processed['CBPredictions'] = seed_avg_test_predictions
        oof_score = mean_squared_error(y_train, df_train_processed['CBPredictions'], squared=False)
        print(f'{"-" * 30}\nCB OOF RMSE: {oof_score:.6} ({len(self.seeds)} Seed Average)\n{"-" * 30}')
                
        self._plot_importance(seed_avg_importance)
        self._plot_predictions(df_train_processed[target], df_train_processed['CBPredictions'], df_test_processed['CBPredictions'])
        
    def _train_and_predict_xgb(self, X_train, y_train, X_test):
        
        seed_avg_oof_predictions = np.zeros(X_train.shape[0])
        seed_avg_test_predictions = np.zeros(X_test.shape[0])
        seed_avg_importance = pd.DataFrame(data=np.zeros(len(self.predictors)), index=self.predictors, columns=['Importance'])
        
        for seed in self.seeds:
            print(f'{"-" * 30}\nRunning XGBoost model with seed: {seed}\n{"-" * 30}\n')
            self.model_parameters['seed'] = seed
        
            for fold in sorted(X_train['fold'].unique()):

                trn_idx, val_idx = X_train.loc[X_train['fold'] != fold].index, X_train.loc[X_train['fold'] == fold].index
                trn = xgb.DMatrix(X_train.loc[trn_idx, self.predictors], label=y_train.loc[trn_idx])
                val = xgb.DMatrix(X_train.loc[val_idx, self.predictors], label=y_train.loc[val_idx])

                model = xgb.train(params=self.model_parameters,
                                  dtrain=trn,
                                  evals=[(trn, 'train'), (val, 'val')],
                                  num_boost_round=self.boosting_rounds, 
                                  early_stopping_rounds=self.early_stopping_rounds,
                                  verbose_eval=500) 

                val_predictions = model.predict(xgb.DMatrix(X_train.loc[val_idx, self.predictors]))
                seed_avg_oof_predictions[val_idx] += (val_predictions / len(self.seeds))
                test_predictions = model.predict(xgb.DMatrix(X_test[self.predictors]))
                seed_avg_test_predictions += (test_predictions / X_train['fold'].nunique() / len(self.seeds))
                seed_avg_importance['Importance'] += (np.array(list(model.get_score(importance_type='gain').values())) / X_train['fold'].nunique() / len(self.seeds))

                fold_score = mean_squared_error(df_train_processed.loc[val_idx, self.target], val_predictions, squared=False)
                print(f'\nXGB Fold {int(fold)} - X_trn: {X_train.loc[trn_idx, self.predictors].shape} X_val: {X_train.loc[val_idx, self.predictors].shape} - Score: {fold_score:.6} - Seed: {seed}\n')
            
        df_train_processed['XGBPredictions'] = seed_avg_oof_predictions
        df_test_processed['XGBPredictions'] = seed_avg_test_predictions
        oof_score = mean_squared_error(y_train, df_train_processed['XGBPredictions'], squared=False)
        print(f'{"-" * 30}\nXGB OOF RMSE: {oof_score:.6} ({len(self.seeds)} Seed Average) \n{"-" * 30}')
                
        self._plot_importance(seed_avg_importance)
        self._plot_predictions(df_train_processed[target], df_train_processed['XGBPredictions'], df_test_processed['XGBPredictions'])
        
    def _train_and_predict_rf(self, X_train, y_train, X_test):
        
        seed_avg_oof_predictions = np.zeros(X_train.shape[0])
        seed_avg_test_predictions = np.zeros(X_test.shape[0])
        
        for seed in self.seeds:
            print(f'{"-" * 30}\nRunning RandomForest model with seed: {seed}\n{"-" * 30}\n')
            self.model_parameters['random_state'] = seed
                
            for fold in sorted(X_train['fold'].unique()):

                trn_idx, val_idx = X_train.loc[X_train['fold'] != fold].index, X_train.loc[X_train['fold'] == fold].index
                X_trn, y_trn = X_train.loc[trn_idx, self.predictors].astype(np.float32), y_train.loc[trn_idx].astype(np.float32)
                X_val, y_val = X_train.loc[val_idx, self.predictors].astype(np.float32), y_train.loc[val_idx].astype(np.float32)

                import cuml
                model = cuml.ensemble.RandomForestRegressor(**self.model_parameters)
                model.fit(X_trn, y_trn)

                val_predictions = model.predict(X_val)
                seed_avg_oof_predictions[val_idx] += (val_predictions / len(self.seeds))
                test_predictions = model.predict(X_test[self.predictors])
                seed_avg_test_predictions += (test_predictions / X_train['fold'].nunique() / len(self.seeds))

                fold_score = mean_squared_error(df_train_processed.loc[val_idx, self.target], val_predictions, squared=False)
                print(f'RF Fold {int(fold)} - X_trn: {X_train.loc[trn_idx, self.predictors].shape} X_val: {X_train.loc[val_idx, self.predictors].shape} - Score: {fold_score:.6}')

        df_train_processed['RFPredictions'] = seed_avg_oof_predictions
        df_test_processed['RFPredictions'] = seed_avg_test_predictions
        oof_score = mean_squared_error(y_train, df_train_processed['RFPredictions'], squared=False)
        print(f'{"-" * 30}\nRF OOF RMSE: {oof_score:.6} ({len(self.seeds)} Seed Average) \n{"-" * 30}')
        
        self._plot_predictions(df_train_processed[target], df_train_processed['RFPredictions'], df_test_processed['RFPredictions'])
        
    def _plot_importance(self, df_importance):
        
        df_importance.sort_values(by='Importance', inplace=True, ascending=False)
        
        plt.figure(figsize=(25, 6))       
        sns.barplot(x='Importance', y=df_importance.index, data=df_importance, palette='Blues_d')
        plt.xlabel('')
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.title(f'{self.model} Feature Importance (Gain)', size=20, pad=20)
        plt.show()
        
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
        
        if self.model == 'LGB':
            self._train_and_predict_lgb(X_train, y_train, X_test)
        elif self.model == 'CB':
            self._train_and_predict_cb(X_train, y_train, X_test)
        elif self.model == 'XGB':
            self._train_and_predict_xgb(X_train, y_train, X_test)
        elif self.model == 'RF':
            self._train_and_predict_rf(X_train, y_train, X_test)