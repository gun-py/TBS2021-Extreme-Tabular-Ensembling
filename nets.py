class NeuralNetworks:
    
    def __init__(self, predictors, target, model, model_parameters, seeds):
        
        self.predictors = predictors
        self.target = target
               
        self.model = model
        self.model_parameters = model_parameters
        self.seeds = seeds
                
    def _set_seed(self, seed):
        
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
                
    def _rmse_loss(self, y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_true - y_pred)))
        
    def _dense_block(self, x, units, activation, dropout_rate, kernel_regularizer, batch_normalization, weight_normalization, gaussian_noise):
        
        if weight_normalization:
            x = tfa.layers.WeightNormalization(Dense(units=units, activation=activation, kernel_regularizer=kernel_regularizer))(x)
        else:
            x = Dense(units=units, activation=activation, kernel_regularizer=kernel_regularizer)(x)
            
        if batch_normalization:
            x = BatchNormalization()(x)
        if dropout_rate > 0:
            x = Dropout(rate=dropout_rate)(x)
        if gaussian_noise > 0:
            x = GaussianNoise(gaussian_noise)(x)
            
        return x
    
    def _get_tmlp(self, input_shape, output_shape):        
        
        inputs = []
        
        cont1_discretized_input = Input(shape=(1, ), name='cont1_discretized_input')
        cont1_embedding_dim = 2
        cont1_discretized_embedding = Embedding(input_dim=4, output_dim=cont1_embedding_dim, input_length=1, name='cont1_discretized_embedding')(cont1_discretized_input)
        cont1_discretized_embedding = Reshape(target_shape=(cont1_embedding_dim, ))(cont1_discretized_embedding)
        inputs.append(cont1_discretized_embedding)
        
        cont2_discretized_input = Input(shape=(1, ), name='cont2_discretized_input')
        cont2_embedding_dim = 4
        cont2_discretized_embedding = Embedding(input_dim=10, output_dim=cont2_embedding_dim, input_length=1, name='cont2_discretized_embedding')(cont2_discretized_input)
        cont2_discretized_embedding = Reshape(target_shape=(cont2_embedding_dim, ))(cont2_discretized_embedding)
        inputs.append(cont2_discretized_embedding)
        
        cont3_discretized_input = Input(shape=(1, ), name='cont3_discretized_input')
        cont3_embedding_dim = 3
        cont3_discretized_embedding = Embedding(input_dim=6, output_dim=cont3_embedding_dim, input_length=1, name='cont3_discretized_embedding')(cont3_discretized_input)
        cont3_discretized_embedding = Reshape(target_shape=(cont3_embedding_dim, ))(cont3_discretized_embedding)
        inputs.append(cont3_discretized_embedding)
        
        cont4_discretized_input = Input(shape=(1, ), name='cont4_discretized_input')
        cont4_embedding_dim = 2
        cont4_discretized_embedding = Embedding(input_dim=4, output_dim=cont4_embedding_dim, input_length=1, name='cont4_discretized_embedding')(cont4_discretized_input)
        cont4_discretized_embedding = Reshape(target_shape=(cont4_embedding_dim, ))(cont4_discretized_embedding)
        inputs.append(cont4_discretized_embedding)
        
        cont5_discretized_input = Input(shape=(1, ), name='cont5_discretized_input')
        cont5_embedding_dim = 2
        cont5_discretized_embedding = Embedding(input_dim=3, output_dim=cont5_embedding_dim, input_length=1, name='cont5_discretized_embedding')(cont5_discretized_input)
        cont5_discretized_embedding = Reshape(target_shape=(cont5_embedding_dim, ))(cont5_discretized_embedding)
        inputs.append(cont5_discretized_embedding)
        
        cont6_discretized_input = Input(shape=(1, ), name='cont6_discretized_input')
        cont6_embedding_dim = 2
        cont6_discretized_embedding = Embedding(input_dim=2, output_dim=cont6_embedding_dim, input_length=1, name='cont6_discretized_embedding')(cont6_discretized_input)
        cont6_discretized_embedding = Reshape(target_shape=(cont6_embedding_dim, ))(cont6_discretized_embedding)
        inputs.append(cont6_discretized_embedding)
        
        cont7_discretized_input = Input(shape=(1, ), name='cont7_discretized_input')
        cont7_embedding_dim = 2
        cont7_discretized_embedding = Embedding(input_dim=3, output_dim=cont7_embedding_dim, input_length=1, name='cont7_discretized_embedding')(cont7_discretized_input)
        cont7_discretized_embedding = Reshape(target_shape=(cont7_embedding_dim, ))(cont7_discretized_embedding)
        inputs.append(cont7_discretized_embedding)
        
        cont8_discretized_input = Input(shape=(1, ), name='cont8_discretized_input')
        cont8_embedding_dim = 2
        cont8_discretized_embedding = Embedding(input_dim=4, output_dim=cont8_embedding_dim, input_length=1, name='cont8_discretized_embedding')(cont8_discretized_input)
        cont8_discretized_embedding = Reshape(target_shape=(cont8_embedding_dim, ))(cont8_discretized_embedding)
        inputs.append(cont8_discretized_embedding)
        
        cont9_discretized_input = Input(shape=(1, ), name='cont9_discretized_input')
        cont9_embedding_dim = 2
        cont9_discretized_embedding = Embedding(input_dim=4, output_dim=cont9_embedding_dim, input_length=1, name='cont9_discretized_embedding')(cont9_discretized_input)
        cont9_discretized_embedding = Reshape(target_shape=(cont9_embedding_dim, ))(cont9_discretized_embedding)
        inputs.append(cont9_discretized_embedding)
        
        cont10_discretized_input = Input(shape=(1, ), name='cont10_discretized_input')
        cont10_embedding_dim = 3
        cont10_discretized_embedding = Embedding(input_dim=8, output_dim=cont10_embedding_dim, input_length=1, name='cont10_discretized_embedding')(cont10_discretized_input)
        cont10_discretized_embedding = Reshape(target_shape=(cont10_embedding_dim, ))(cont10_discretized_embedding)
        inputs.append(cont10_discretized_embedding)
        
        cont11_discretized_input = Input(shape=(1, ), name='cont11_discretized_input')
        cont11_embedding_dim = 3
        cont11_discretized_embedding = Embedding(input_dim=5, output_dim=cont11_embedding_dim, input_length=1, name='cont11_discretized_embedding')(cont11_discretized_input)
        cont11_discretized_embedding = Reshape(target_shape=(cont11_embedding_dim, ))(cont11_discretized_embedding)
        inputs.append(cont11_discretized_embedding)
        
        cont12_discretized_input = Input(shape=(1, ), name='cont12_discretized_input')
        cont12_embedding_dim = 2
        cont12_discretized_embedding = Embedding(input_dim=4, output_dim=cont12_embedding_dim, input_length=1, name='cont12_discretized_embedding')(cont12_discretized_input)
        cont12_discretized_embedding = Reshape(target_shape=(cont12_embedding_dim, ))(cont12_discretized_embedding)
        inputs.append(cont12_discretized_embedding)
        
        cont13_discretized_input = Input(shape=(1, ), name='cont13_discretized_input')
        cont13_embedding_dim = 3
        cont13_discretized_embedding = Embedding(input_dim=6, output_dim=cont13_embedding_dim, input_length=1, name='cont13_discretized_embedding')(cont13_discretized_input)
        cont13_discretized_embedding = Reshape(target_shape=(cont13_embedding_dim, ))(cont13_discretized_embedding)
        inputs.append(cont13_discretized_embedding)
        
        cont14_discretized_input = Input(shape=(1, ), name='cont14_discretized_input')
        cont14_embedding_dim = 3
        cont14_discretized_embedding = Embedding(input_dim=6, output_dim=cont14_embedding_dim, input_length=1, name='cont14_discretized_embedding')(cont14_discretized_input)
        cont14_discretized_embedding = Reshape(target_shape=(cont14_embedding_dim, ))(cont14_discretized_embedding)
        inputs.append(cont14_discretized_embedding)
        
        continuous_inputs = Input(shape=(14, ), name='continuous_inputs')
        inputs.append(continuous_inputs)
        
        sparse_inputs = Input(shape=(input_shape - (14 * 2), ), name='sparse_inputs')
        inputs.append(sparse_inputs)
        
        x = Concatenate()(inputs)
        x = self._dense_block(x, units=256, activation='elu', dropout_rate=0.50, kernel_regularizer=None, batch_normalization=True, weight_normalization=True, gaussian_noise=0.01)        
        x = self._dense_block(x, units=256, activation='elu', dropout_rate=0.50, kernel_regularizer=None, batch_normalization=True, weight_normalization=True, gaussian_noise=0.01)        
        outputs = Dense(output_shape, activation='linear')(x)
        
        model = Model(inputs=[continuous_inputs,
                              cont1_discretized_input,
                              cont2_discretized_input,
                              cont3_discretized_input,
                              cont4_discretized_input,
                              cont5_discretized_input,
                              cont6_discretized_input,
                              cont7_discretized_input,
                              cont8_discretized_input,
                              cont9_discretized_input,
                              cont10_discretized_input,
                              cont11_discretized_input,
                              cont12_discretized_input,
                              cont13_discretized_input,
                              cont14_discretized_input, 
                              sparse_inputs],
                      outputs=outputs,
                      name='TMLP')                
        
        optimizer = tfa.optimizers.AdamW(learning_rate=self.model_parameters['learning_rate'], weight_decay=self.model_parameters['weight_decay'])          
        model.compile(optimizer=optimizer, loss=self._rmse_loss, metrics=None)
        
        return model
    
    def _get_rmlp(self, input_shape, output_shape):
        
        inputs = Input(shape=(input_shape, ), name='inputs')
        x = self._dense_block(inputs, units=128, activation='elu', dropout_rate=0.30, kernel_regularizer=None, batch_normalization=True, weight_normalization=False, gaussian_noise=0)        
        x = self._dense_block(x, units=128, activation='elu', dropout_rate=0.30, kernel_regularizer=None, batch_normalization=True, weight_normalization=False, gaussian_noise=0)        
        outputs = Dense(output_shape, activation='linear')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='RMLP')                
        
        optimizer = Adam(learning_rate=self.model_parameters['learning_rate'])          
        model.compile(optimizer=optimizer, loss=self._rmse_loss, metrics=None)
        
        return model
        
    def train_and_predict_tmlp(self, X_train, y_train, X_test):
        
        all_model_histories = {}
        seed_avg_oof_predictions = np.zeros(X_train.shape[0])
        seed_avg_test_predictions = np.zeros(X_test.shape[0])
                                        
        oof_predictions = np.zeros(X_train.shape[0])
        test_predictions = np.zeros(X_test.shape[0])
        
        for seed in self.seeds:
            
            X = pd.concat([X_train[continuous_features], X_test[continuous_features]], ignore_index=True, axis=0)
            embedder = RandomTreesEmbedding(n_estimators=300,
                                            max_depth=3,
                                            min_samples_split=100,
                                            n_jobs=-1,
                                            random_state=seed, 
                                            verbose=True)
            embedder.fit(X)        
            del X

            train_sparse_features = embedder.transform(X_train.loc[:, continuous_features]).astype(np.uint8).toarray()
            X_train = pd.concat([X_train, pd.DataFrame(train_sparse_features)], axis=1)
            sparse_feature_columns = list(np.arange(train_sparse_features.shape[1]))
            self.predictors += sparse_feature_columns
            del train_sparse_features

            test_sparse_features = embedder.transform(X_test.loc[:, continuous_features]).astype(np.uint8).toarray()
            del embedder
            X_test = pd.concat([X_test, pd.DataFrame(test_sparse_features)], axis=1)
            del test_sparse_features

            print(f'{"-" * 30}\nRunning TMLP model with seed: {seed}\n{"-" * 30}\n')
            self._set_seed(seed)
            model_histories = []
        
            for fold in sorted(X_train['fold'].unique()):

                trn_idx, val_idx = X_train.loc[X_train['fold'] != fold].index, X_train.loc[X_train['fold'] == fold].index
                X_trn, y_trn = X_train.loc[trn_idx, self.predictors], y_train.loc[trn_idx]
                X_val, y_val = X_train.loc[val_idx, self.predictors], y_train.loc[val_idx]

                model = self._get_mlp(input_shape=X_trn.shape[1], output_shape=1)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                              factor=self.model_parameters['reduce_lr_factor'],
                                              patience=self.model_parameters['reduce_lr_patience'],
                                              min_lr=self.model_parameters['reduce_lr_min'],
                                              mode='min',
                                              verbose=True)
                early_stopping = EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               min_delta=self.model_parameters['early_stopping_min_delta'],
                                               patience=self.model_parameters['early_stopping_patience'],
                                               restore_best_weights=True, 
                                               verbose=True)
                
                history = model.fit([X_trn[self.predictors[:14]],
                                     X_trn['cont1_class'],
                                     X_trn['cont2_class'],
                                     X_trn['cont3_class'],
                                     X_trn['cont4_class'],
                                     X_trn['cont5_class'],
                                     X_trn['cont6_class'],
                                     X_trn['cont7_class'],
                                     X_trn['cont8_class'],
                                     X_trn['cont9_class'],
                                     X_trn['cont10_class'],
                                     X_trn['cont11_class'],
                                     X_trn['cont12_class'],
                                     X_trn['cont13_class'],
                                     X_trn['cont14_class'], 
                                     X_trn[sparse_feature_columns]],
                                    y_trn,
                                    validation_data=([X_val[self.predictors[:14]],
                                                      X_val['cont1_class'],
                                                      X_val['cont2_class'],
                                                      X_val['cont3_class'],
                                                      X_val['cont4_class'],
                                                      X_val['cont5_class'],
                                                      X_val['cont6_class'],
                                                      X_val['cont7_class'],
                                                      X_val['cont8_class'],
                                                      X_val['cont9_class'],
                                                      X_val['cont10_class'],
                                                      X_val['cont11_class'],
                                                      X_val['cont12_class'],
                                                      X_val['cont13_class'],
                                                      X_val['cont14_class'], 
                                                      X_val[sparse_feature_columns]],
                                                     y_val),
                                    epochs=self.model_parameters['epochs'], 
                                    batch_size=self.model_parameters['batch_size'],
                                    callbacks=[reduce_lr, early_stopping],
                                    verbose=True)

                val_predictions = model.predict([X_val[self.predictors[:14]],
                                                 X_val['cont1_class'],
                                                 X_val['cont2_class'],
                                                 X_val['cont3_class'],
                                                 X_val['cont4_class'],
                                                 X_val['cont5_class'],
                                                 X_val['cont6_class'],
                                                 X_val['cont7_class'],
                                                 X_val['cont8_class'],
                                                 X_val['cont9_class'],
                                                 X_val['cont10_class'],
                                                 X_val['cont11_class'],
                                                 X_val['cont12_class'],
                                                 X_val['cont13_class'],
                                                 X_val['cont14_class'],
                                                 X_val[sparse_feature_columns]])
                seed_avg_oof_predictions[val_idx] += (val_predictions.flatten() / len(self.seeds))
                test_predictions = model.predict([X_test[self.predictors[:14]],
                                                  X_test['cont1_class'],
                                                  X_test['cont2_class'],
                                                  X_test['cont3_class'],
                                                  X_test['cont4_class'],
                                                  X_test['cont5_class'],
                                                  X_test['cont6_class'],
                                                  X_test['cont7_class'],
                                                  X_test['cont8_class'],
                                                  X_test['cont9_class'],
                                                  X_test['cont10_class'],
                                                  X_test['cont11_class'],
                                                  X_test['cont12_class'],
                                                  X_test['cont13_class'],
                                                  X_test['cont14_class'],
                                                  X_test[sparse_feature_columns]])
                seed_avg_test_predictions += (test_predictions.flatten() / X_train['fold'].nunique() / len(self.seeds))
                model_histories.append(history)

                fold_score = mean_squared_error(df_train_processed.loc[val_idx, self.target], val_predictions, squared=False)
                print(f'\nTMLP Fold {int(fold)} - X_trn: {X_train.loc[trn_idx, self.predictors].shape} X_val: {X_train.loc[val_idx, self.predictors].shape} - Score: {fold_score:.6} - Seed: {seed}\n')
            
            all_model_histories[seed] = model_histories
            
        df_train_processed['TMLPPredictions'] = seed_avg_oof_predictions
        df_test_processed['TMLPPredictions'] = seed_avg_test_predictions
        oof_score = mean_squared_error(y_train, df_train_processed['TMLPPredictions'], squared=False)
        print(f'{"-" * 30}\nTMLP OOF RMSE: {oof_score:.6} ({len(self.seeds)} Seed Average) \n{"-" * 30}')
        
        self._plot_learning_curve(all_model_histories)
        self._plot_predictions(df_train_processed[target], df_train_processed['TMLPPredictions'], df_test_processed['TMLPPredictions'])
        
    def train_and_predict_rmlp(self, X_train, y_train, X_test):
        
        all_model_histories = {}
        seed_avg_oof_predictions = np.zeros(X_train.shape[0])
        seed_avg_test_predictions = np.zeros(X_test.shape[0])
                                        
        oof_predictions = np.zeros(X_train.shape[0])
        test_predictions = np.zeros(X_test.shape[0])
        
        for seed in self.seeds:
            
            print(f'{"-" * 30}\nRunning RMLP model with seed: {seed}\n{"-" * 30}\n')
            self._set_seed(seed)
            model_histories = []
        
            for fold in sorted(X_train['fold'].unique()):

                trn_idx, val_idx = X_train.loc[X_train['fold'] != fold].index, X_train.loc[X_train['fold'] == fold].index
                X_trn, y_trn = X_train.loc[trn_idx, self.predictors], y_train.loc[trn_idx]
                X_val, y_val = X_train.loc[val_idx, self.predictors], y_train.loc[val_idx]

                model = self._get_rmlp(input_shape=X_trn.shape[1], output_shape=1)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                              factor=self.model_parameters['reduce_lr_factor'],
                                              patience=self.model_parameters['reduce_lr_patience'],
                                              min_lr=self.model_parameters['reduce_lr_min'],
                                              mode='min',
                                              verbose=True)
                early_stopping = EarlyStopping(monitor='val_loss',
                                               mode='min',
                                               min_delta=self.model_parameters['early_stopping_min_delta'],
                                               patience=self.model_parameters['early_stopping_patience'],
                                               restore_best_weights=True, 
                                               verbose=True)
                
                history = model.fit(X_trn, y_trn,
                                    validation_data=(X_val, y_val),
                                    epochs=self.model_parameters['epochs'], 
                                    batch_size=self.model_parameters['batch_size'],
                                    callbacks=[reduce_lr, early_stopping],
                                    verbose=True)

                val_predictions = model.predict(X_val)
                seed_avg_oof_predictions[val_idx] += (val_predictions.flatten() / len(self.seeds))
                test_predictions = model.predict(X_test)
                seed_avg_test_predictions += (test_predictions.flatten() / X_train['fold'].nunique() / len(self.seeds))
                model_histories.append(history)

                fold_score = mean_squared_error(df_train_processed.loc[val_idx, self.target], val_predictions.flatten(), squared=False)
                print(f'\nRMLP Fold {int(fold)} - X_trn: {X_train.loc[trn_idx, self.predictors].shape} X_val: {X_train.loc[val_idx, self.predictors].shape} - Score: {fold_score:.6} - Seed: {seed}\n')
            
            all_model_histories[seed] = model_histories
            
        df_train_processed['RMLPPredictions'] = seed_avg_oof_predictions
        df_test_processed['RMLPPredictions'] = seed_avg_test_predictions
        oof_score = mean_squared_error(y_train, df_train_processed['RMLPPredictions'], squared=False)
        print(f'{"-" * 30}\nRMLP OOF RMSE: {oof_score:.6} ({len(self.seeds)} Seed Average) \n{"-" * 30}')
        
        self._plot_learning_curve(all_model_histories)
        self._plot_predictions(df_train_processed[target], df_train_processed['RMLPPredictions'], df_test_processed['RMLPPredictions'])
        
    def _plot_learning_curve(self, all_model_histories):
    
        n_folds = 5
        fig, axes = plt.subplots(nrows=n_folds, figsize=(32, 50), dpi=100)

        for i in range(n_folds):

            for seed, histories in all_model_histories.items():

                axes[i].plot(np.arange(1, len(histories[i].history['loss']) + 1), histories[i].history['loss'], label=f'train_loss (Seed: {seed})', alpha=0.5)
                axes[i].plot(np.arange(1, len(histories[i].history['val_loss']) + 1), histories[i].history['val_loss'], label=f'val_loss (Seed: {seed})', alpha=0.5)    

            axes[i].set_xlabel('Epochs', size=20)
            axes[i].set_ylabel('RMSE', size=20)
            axes[i].set_yscale('log')
            axes[i].tick_params(axis='x', labelsize=20)
            axes[i].tick_params(axis='y', labelsize=20)
            axes[i].legend(prop={'size': 20}) 
            axes[i].set_title(f'Fold {i + 1} Learning Curve', fontsize=20, pad=10)

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