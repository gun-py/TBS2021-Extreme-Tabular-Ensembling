class Preprocessor:
    
    def __init__(self, train, test, n_splits, shuffle, random_state, scaler, discretize_features, create_features):
        self.train = train.copy(deep=True)        
        self.test = test.copy(deep=True)   
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.scaler = scaler() if scaler else None
        self.discretize_features = discretize_features
        self.create_features = create_features
        
    def drop_outliers(self):
        outlier_idx = self.train[self.train[target] < 4].index
        self.train.drop(outlier_idx, inplace=True)
        self.train.reset_index(drop=True, inplace=True) 
        print(f'Dropped {len(outlier_idx)} outliers')
        del outlier_idx
        
    def create_folds(self):       
        kf = KFold(n_splits=self.n_splits, shuffle=self.shuffle, random_state=self.random_state)
        for fold, (_, val_idx) in enumerate(kf.split(self.train), 1):
            self.train.loc[val_idx, 'fold'] = fold
        self.train['fold'] = self.train['fold'].astype(np.uint8)
                
    def scale(self):
        df_all = pd.concat([self.train[continuous_features], self.test[continuous_features]], ignore_index=True, axis=0)
        self.scaler.fit(df_all[continuous_features])
        self.train.loc[:, continuous_features] = self.scaler.transform(self.train.loc[:, continuous_features].values)
        self.test.loc[:, continuous_features] = self.scaler.transform(self.test.loc[:, continuous_features].values)
        print(f'Scaled {len(continuous_features)} features with {self.scaler.__class__.__name__}')
        del df_all
        
    def create_peak_features(self):
        for df in [self.train, self.test]:
            df['cont2_peak'] = ((df_train['cont2'].round(2) == 0.36) | (df_train['cont2'].round(2) == 0.42) | (df_train['cont2'].round(2) == 0.49) |\
                                (df_train['cont2'].round(2) == 0.55) | (df_train['cont2'].round(2) == 0.56) | (df_train['cont2'].round(2) == 0.62) |\
                                (df_train['cont2'].round(2) == 0.68) | (df_train['cont2'].round(2) == 0.74)).astype(np.uint8)
            
            df['cont5_peak'] = (df_train['cont5'].round(2) == 0.28).astype(np.uint8)
            
        peak_features = ['cont2_peak', 'cont5_peak']
        print(f'Created {len(peak_features)} peak features')
            
    def create_idx_features(self):
        for df in [self.train, self.test]:
            df['cont_argmin'] = np.argmin(df[continuous_features].values, axis=1)
            df['cont_argmax'] = np.argmax(df[continuous_features].values, axis=1)
            
        idx_features = ['cont_argmin', 'cont_argmax']
        print(f'Created {len(idx_features)} idx features with argmin and argmax')
            
    def create_gmm_features(self):
        n_component_mapping = {
            1: 4,
            2: 10,
            3: 6,
            4: 4,
            5: 3,
            6: 2,
            7: 3,
            8: 4,
            9: 4,
            10: 8,
            11: 5,
            12: 4,
            13: 6,
            14: 6
        }
        
        for i in range(1, 15):
            gmm = GaussianMixture(n_components=n_component_mapping[i], random_state=self.random_state)            
            gmm.fit(pd.concat([self.train[f'cont{i}'], self.test[f'cont{i}']], axis=0).values.reshape(-1, 1))
            
            self.train[f'cont{i}_class'] = gmm.predict(self.train[f'cont{i}'].values.reshape(-1, 1))
            self.test[f'cont{i}_class'] = gmm.predict(self.test[f'cont{i}'].values.reshape(-1, 1))
            
        gmm_features = [f'cont{i}_class' for i in range(1, 15)]
        print(f'Created {len(gmm_features)} gmm features with GaussianMixture')
                
    def transform(self):
        self.drop_outliers()
        self.create_folds()
        
        if self.create_features:
            self.create_peak_features()
            self.create_idx_features()
            
        if self.discretize_features:
            self.create_gmm_features()
                    
        if self.scaler:
            self.scale()
        
        return self.train.copy(deep=True), self.test.copy(deep=True)

cross_validation_seed = 0
preprocessor = Preprocessor(train=df_train, test=df_test,
                            n_splits=5, shuffle=True, random_state=cross_validation_seed,
                            scaler=None,
                            create_features=False, discretize_features=False)
df_train_processed, df_test_processed = preprocessor.transform()

print(f'\nPreprocessed Training Set Shape = {df_train_processed.shape}')
print(f'Preprocessed Training Set Memory Usage = {df_train_processed.memory_usage().sum() / 1024 ** 2:.2f} MB')
print(f'Preprocessed Test Set Shape = {df_test_processed.shape}')
print(f'Preprocessed Test Set Memory Usage = {df_test_processed.memory_usage().sum() / 1024 ** 2:.2f} MB\n')