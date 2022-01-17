import warnings
warnings.filterwarnings('ignore')
import random
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 50)

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import probplot, kurtosis, skew, gmean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomTreesEmbedding

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Embedding, Reshape, Concatenate, Dropout, BatchNormalization, Activation, GaussianNoise
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
continuous_features = [feature for feature in df_train.columns if feature.startswith('cont')]
target = 'target'

print(f'Training Set Shape = {df_train.shape}')
print(f'Training Set Memory Usage = {df_train.memory_usage().sum() / 1024 ** 2:.2f} MB')
print(f'Test Set Shape = {df_test.shape}')
print(f'Test Set Memory Usage = {df_test.memory_usage().sum() / 1024 ** 2:.2f} MB')

TRAIN_LGB = True
TRAIN_CB = True
TRAIN_XGB = True
TRAIN_RF = True
FIT_RR = True
FIT_SVM = True
TRAIN_TMLP = True
TRAIN_RMLP = True

if TRAIN_LGB:
    model = 'LGB'
    lgb_preprocessor = Preprocessor(train=df_train, test=df_test,
                                    n_splits=5, shuffle=True, random_state=cross_validation_seed, scaler=None,
                                    create_features=False, discretize_features=False)
    df_train_lgb, df_test_lgb = lgb_preprocessor.transform()

    print(f'\n{model} Training Set Shape = {df_train_lgb.shape}')
    print(f'{model} Training Set Memory Usage = {df_train_lgb.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'{model} Test Set Shape = {df_test_lgb.shape}')
    print(f'{model} Test Set Memory Usage = {df_test_lgb.memory_usage().sum() / 1024 ** 2:.2f} MB\n')

    X_train_lgb = df_train_lgb.copy(deep=True)
    y_train_lgb = df_train_lgb[target].copy(deep=True)
    X_test_lgb = df_test_lgb.copy(deep=True)

    lgb_parameters = {
        'predictors': continuous_features,
        'target': target,
        'model': model,
        'model_parameters': {
            'num_leaves': 2 ** 8, 
            'learning_rate': 0.001,
            'bagging_fraction': 0.6,
            'bagging_freq': 3,
            'feature_fraction': 0.5,
            'feature_fraction_bynode': 0.8,
            'min_data_in_leaf': 100,
            'min_data_per_group': 1,            
            'min_gain_to_split': 0.001,
            'lambda_l1': 6,
            'lambda_l2': 0.0005,
            'max_bin': 768,
            'max_depth': -1,
            'objective': 'regression',
            'seed': None,
            'feature_fraction_seed': None,
            'bagging_seed': None,
            'drop_seed': None,
            'data_random_seed': None,
            'boosting_type': 'gbdt',
            'verbose': 1,
            'metric': 'rmse',
            'n_jobs': -1,
        },
        'boosting_rounds': 20000,
        'early_stopping_rounds': 200,
        'seeds': [541992, 721991, 1337]
    }

    lgb_model = TreeModels(**lgb_parameters)
    lgb_model.run(X_train_lgb, y_train_lgb, X_test_lgb)

    del df_train_lgb, df_test_lgb, X_train_lgb, y_train_lgb, X_test_lgb
    del lgb_preprocessor, lgb_parameters, lgb_model
    
    print('Saving LightGBM OOF and Test predictions to current working directory.')
    df_train_processed[['id', 'LGBPredictions']].to_csv('lgb_oof_predictions.csv', index=False)
    df_test_processed[['id', 'LGBPredictions']].to_csv('lgb_test_predictions.csv', index=False)
    
else:
    print('Loading LightGBM OOF and Test predictions from current working directory.')
    df_train_processed['LGBPredictions'] = pd.read_csv('lgb_oof_predictions.csv')['LGBPredictions']
    df_test_processed['LGBPredictions'] = pd.read_csv('lgb_test_predictions.csv')['LGBPredictions']    
    oof_score = mean_squared_error(df_train_processed['target'], df_train_processed['LGBPredictions'], squared=False)
    print(f'LGB OOF RMSE: {oof_score:.6}')
    TreeModels._plot_predictions(None, df_train_processed[target], df_train_processed['LGBPredictions'], df_test_processed['LGBPredictions'])
    

if TRAIN_CB:
    model = 'CB'
    cb_preprocessor = Preprocessor(train=df_train, test=df_test,
                                   n_splits=5, shuffle=True, random_state=cross_validation_seed, scaler=None,
                                   create_features=False, discretize_features=False)
    df_train_cb, df_test_cb = cb_preprocessor.transform()

    print(f'\n{model} Training Set Shape = {df_train_cb.shape}')
    print(f'{model} Training Set Memory Usage = {df_train_cb.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'{model} Test Set Shape = {df_test_cb.shape}')
    print(f'{model} Test Set Memory Usage = {df_test_cb.memory_usage().sum() / 1024 ** 2:.2f} MB\n')

    X_train_cb = df_train_cb[continuous_features + ['fold']].copy(deep=True)
    y_train_cb = df_train_cb[target].copy(deep=True)
    X_test_cb = df_test_cb[continuous_features].copy(deep=True)

    cb_parameters = {
        'predictors': continuous_features,
        'target': target,
        'model': model,
        'model_parameters': {
            'n_estimators': 20000, 
            'learning_rate': 0.006,
            'depth': 10,
            'subsample': 0.8,
            'colsample_bylevel': 0.5,
            'l2_leaf_reg': 0.1,
            'metric_period': 1000,
            'boost_from_average': True,
            'use_best_model': True,
            'eval_metric': 'RMSE',
            'loss_function': 'RMSE',   
            'od_type': 'Iter',
            'od_wait': 200,
            'random_seed': None,
            'verbose': 1,
        },
        'boosting_rounds': None,
        'early_stopping_rounds': None,
        'seeds': [541992, 721991, 1337, 42, 0]
    }

    cb_model = TreeModels(**cb_parameters)
    cb_model.run(X_train_cb, y_train_cb, X_test_cb)
    
    del df_train_cb, df_test_cb, X_train_cb, y_train_cb, X_test_cb
    del cb_preprocessor, cb_parameters, cb_model
    
    print('Saving CatBoost OOF and Test predictions to current working directory.')
    df_train_processed[['id', 'CBPredictions']].to_csv('cb_oof_predictions.csv', index=False)
    df_test_processed[['id', 'CBPredictions']].to_csv('cb_test_predictions.csv', index=False)
    
else:
    print('Loading CatBoost OOF and Test predictions from current working directory.')
    df_train_processed['CBPredictions'] = pd.read_csv('cb_oof_predictions.csv')['CBPredictions']
    df_test_processed['CBPredictions'] = pd.read_csv('cb_test_predictions.csv')['CBPredictions']    
    oof_score = mean_squared_error(df_train_processed['target'], df_train_processed['CBPredictions'], squared=False)
    print(f'CB OOF RMSE: {oof_score:.6}')
    TreeModels._plot_predictions(None, df_train_processed[target], df_train_processed['CBPredictions'], df_test_processed['CBPredictions'])


if TRAIN_XGB:
    model = 'XGB'
    xgb_preprocessor = Preprocessor(train=df_train, test=df_test,
                                    n_splits=5, shuffle=True, random_state=cross_validation_seed, scaler=None,
                                    create_features=False, discretize_features=False)
    df_train_xgb, df_test_xgb = xgb_preprocessor.transform()

    print(f'\n{model} Training Set Shape = {df_train_xgb.shape}')
    print(f'{model} Training Set Memory Usage = {df_train_xgb.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'{model} Test Set Shape = {df_test_xgb.shape}')
    print(f'{model} Test Set Memory Usage = {df_test_xgb.memory_usage().sum() / 1024 ** 2:.2f} MB\n')

    X_train_xgb = df_train_xgb[continuous_features + ['fold']].copy(deep=True)
    y_train_xgb = df_train_xgb[target].copy(deep=True)
    X_test_xgb = df_test_xgb[continuous_features].copy(deep=True)

    xgb_parameters = {
        'predictors': continuous_features,
        'target': target,
        'model': model,
        'model_parameters': {
            'learning_rate': 0.002,
            'colsample_bytree': 0.6, 
            'colsample_bylevel': 0.6,
            'colsample_bynode': 0.6,
            'sumbsample': 0.8,
            'max_depth': 14,
            'gamma': 0,
            'min_child_weight': 200,
            'lambda': 0,
            'alpha': 0,
            'objective': 'reg:squarederror',
            'seed': None,
            'boosting_type': 'gbtree',
            'tree_method': 'gpu_hist',
            'silent': True,
            'verbose': 1,
            'n_jobs': -1,
        },
        'boosting_rounds': 25000,
        'early_stopping_rounds': 200,
        'seeds': [541992, 721991, 1337]
    }

    xgb_model = TreeModels(**xgb_parameters)
    xgb_model.run(X_train_xgb, y_train_xgb, X_test_xgb)

    del df_train_xgb, df_test_xgb, X_train_xgb, y_train_xgb, X_test_xgb
    del xgb_preprocessor, xgb_parameters, xgb_model
    
    print('Saving XGBoost OOF and Test predictions to current working directory.')
    df_train_processed[['id', 'XGBPredictions']].to_csv('xgb_oof_predictions.csv', index=False)
    df_test_processed[['id', 'XGBPredictions']].to_csv('xgb_test_predictions.csv', index=False)
    
else:
    print('Loading XGBoost OOF and Test predictions from current working directory.')
    df_train_processed['XGBPredictions'] = pd.read_csv('xgb_oof_predictions.csv')['XGBPredictions']
    df_test_processed['XGBPredictions'] = pd.read_csv('xgb_test_predictions.csv')['XGBPredictions']    
    oof_score = mean_squared_error(df_train_processed['target'], df_train_processed['XGBPredictions'], squared=False)
    print(f'XGB OOF RMSE: {oof_score:.6}')
    TreeModels._plot_predictions(None, df_train_processed[target], df_train_processed['XGBPredictions'], df_test_processed['XGBPredictions'])


if TRAIN_RF:
    model = 'RF'
    rf_preprocessor = Preprocessor(train=df_train, test=df_test,
                                   n_splits=5, shuffle=True, random_state=cross_validation_seed, scaler=None,
                                   create_features=False, discretize_features=False)
    df_train_rf, df_test_rf = rf_preprocessor.transform()

    print(f'\n{model} Training Set Shape = {df_train_rf.shape}')
    print(f'{model} Training Set Memory Usage = {df_train_rf.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'{model} Test Set Shape = {df_test_rf.shape}')
    print(f'{model} Test Set Memory Usage = {df_test_rf.memory_usage().sum() / 1024 ** 2:.2f} MB\n')

    X_train_rf = df_train_rf[continuous_features + ['fold']].copy(deep=True)
    y_train_rf = df_train_rf[target].copy(deep=True)
    X_test_rf = df_test_rf[continuous_features].copy(deep=True)

    rf_parameters = {
        'predictors': continuous_features,
        'target': target,
        'model': model,
        'model_parameters': {
            'n_estimators': 400,
            'split_algo': 0,
            'split_criterion': 2,             
            'bootstrap': True,
            'bootstrap_features': False,
            'max_depth': 13,
            'max_leaves': -1,
            'max_features': 0.5,
            'n_bins': 2 ** 6,
            'random_state': None,
            'verbose': True,
        },
        'boosting_rounds': None,
        'early_stopping_rounds': None,
        'seeds': [541992, 721991, 1337, 42, 0]
    }

    rf_model = TreeModels(**rf_parameters)
    rf_model.run(X_train_rf, y_train_rf, X_test_rf)

    del df_train_rf, df_test_rf, X_train_rf, y_train_rf, X_test_rf
    del rf_preprocessor, rf_parameters, rf_model
    
    print('Saving RandomForest OOF and Test predictions to current working directory.')
    df_train_processed[['id', 'RFPredictions']].to_csv('rf_oof_predictions.csv', index=False)
    df_test_processed[['id', 'RFPredictions']].to_csv('rf_test_predictions.csv', index=False)
    
else:
    print('Loading RandomForest OOF and Test predictions from current working directory.')
    df_train_processed['RFPredictions'] = pd.read_csv('rf_oof_predictions.csv')['RFPredictions']
    df_test_processed['RFPredictions'] = pd.read_csv('rf_test_predictions.csv')['RFPredictions']    
    oof_score = mean_squared_error(df_train_processed['target'], df_train_processed['RFPredictions'], squared=False)
    print(f'RF OOF RMSE: {oof_score:.6}')
    TreeModels._plot_predictions(None, df_train_processed[target], df_train_processed['RFPredictions'], df_test_processed['RFPredictions'])


if FIT_RR:
    model = 'Ridge'
    ridge_preprocessor = Preprocessor(train=df_train, test=df_test,
                                      n_splits=5, shuffle=True, random_state=cross_validation_seed, scaler=None,
                                      create_features=False, discretize_features=False)
    df_train_ridge, df_test_ridge = ridge_preprocessor.transform()

    print(f'\n{model} Training Set Shape = {df_train_ridge.shape}')
    print(f'{model} Training Set Memory Usage = {df_train_ridge.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'{model} Test Set Shape = {df_test_ridge.shape}')
    print(f'{model} Test Set Memory Usage = {df_test_ridge.memory_usage().sum() / 1024 ** 2:.2f} MB\n')
    
    X_train_rr = df_train_ridge[continuous_features + ['fold']].copy(deep=True)
    y_train_rr = df_train_ridge[target].copy(deep=True)
    X_test_rr = df_test_ridge[continuous_features].copy(deep=True)
    
    ridge_parameters = {
        'predictors': continuous_features,
        'target': target,
        'model': model,
        'model_parameters': {
            'alpha': 7000
        }
    }

    ridge_model = LinearModels(**ridge_parameters)
    ridge_model.run(X_train_rr, y_train_rr, X_test_rr)

    del df_train_ridge, df_test_ridge, X_train_ridge, y_train_ridge, X_test_ridge
    del ridge_preprocessor, ridge_parameters, ridge_model
    
    print('Saving RidgeRegression OOF and Test predictions to current working directory.')
    df_train_processed[['id', 'RRPredictions']].to_csv('rr_oof_predictions.csv', index=False)
    df_test_processed[['id', 'RRPredictions']].to_csv('rr_test_predictions.csv', index=False)
    
else:
    print('Loading RidgeRegression OOF and Test predictions from current working directory.')
    df_train_processed['RRPredictions'] = pd.read_csv('rr_oof_predictions.csv')['RRPredictions']
    df_test_processed['RRPredictions'] = pd.read_csv('rr_test_predictions.csv')['RRPredictions']    
    oof_score = mean_squared_error(df_train_processed['target'], df_train_processed['RRPredictions'], squared=False)
    print(f'RR OOF RMSE: {oof_score:.6}')
    LinearModels._plot_predictions(None, df_train_processed[target], df_train_processed['RRPredictions'], df_test_processed['RRPredictions'])


if FIT_SVM:
    model = 'SVM'
    svm_preprocessor = Preprocessor(train=df_train, test=df_test,
                                    n_splits=5, shuffle=True, random_state=cross_validation_seed, scaler=StandardScaler,
                                    create_features=False, discretize_features=True)
    df_train_svm, df_test_svm = svm_preprocessor.transform()

    print(f'\n{model} Training Set Shape = {df_train_svm.shape}')
    print(f'{model} Training Set Memory Usage = {df_train_svm.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'{model} Test Set Shape = {df_test_svm.shape}')
    print(f'{model} Test Set Memory Usage = {df_test_svm.memory_usage().sum() / 1024 ** 2:.2f} MB\n')

    X_train_svm = df_train_svm[continuous_features + ['fold'] + [f'{cont}_class' for cont in continuous_features]].copy(deep=True)
    y_train_svm = df_train_svm[target].copy(deep=True)
    X_test_svm = df_test_svm[continuous_features + [f'{cont}_class' for cont in continuous_features]].copy(deep=True)
    
    svm_parameters = {
        'predictors': continuous_features + [f'{cont}_class' for cont in continuous_features],
        'target': target,
        'model': model,
        'model_parameters': {
            'C': 0.5
        }
    }

    svm_model = LinearModels(**svm_parameters)
    svm_model.run(X_train_svm, y_train_svm, X_test_svm)

    del df_train_svm, df_test_svm, X_train_svm, y_train_svm, X_test_svm
    del svm_preprocessor, svm_parameters, svm_model
    
    print('Saving SVM OOF and Test predictions to current working directory.')
    df_train_processed[['id', 'SVMPredictions']].to_csv('svm_oof_predictions.csv', index=False)
    df_test_processed[['id', 'SVMPredictions']].to_csv('svm_test_predictions.csv', index=False)
    
else:
    print('Loading SVM OOF and Test predictions from current working directory.')
    df_train_processed['SVMPredictions'] = pd.read_csv('svm_oof_predictions.csv')['SVMPredictions']
    df_test_processed['SVMPredictions'] = pd.read_csv('svm_test_predictions.csv')['SVMPredictions']    
    oof_score = mean_squared_error(df_train_processed['target'], df_train_processed['SVMPredictions'], squared=False)
    print(f'SVM OOF RMSE: {oof_score:.6}')
    LinearModels._plot_predictions(None, df_train_processed[target], df_train_processed['SVMPredictions'], df_test_processed['SVMPredictions'])


if TRAIN_TMLP:
    model = 'TMLP'
    tmlp_preprocessor = Preprocessor(train=df_train, test=df_test,
                                     n_splits=5, shuffle=True, random_state=cross_validation_seed, scaler=StandardScaler,
                                     create_features=False, discretize_features=True)
    df_train_tmlp, df_test_tmlp = tmlp_preprocessor.transform()

    print(f'\n{model} Training Set Shape = {df_train_tmlp.shape}')
    print(f'{model} Training Set Memory Usage = {df_train_tmlp.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'{model} Test Set Shape = {df_test_tmlp.shape}')
    print(f'{model} Test Set Memory Usage = {df_test_tmlp.memory_usage().sum() / 1024 ** 2:.2f} MB\n')

    X_train_tmlp = df_train_tmlp[continuous_features + ['fold'] + [f'{cont}_class' for cont in continuous_features]].copy(deep=True)
    y_train_tmlp = df_train_tmlp[target].copy(deep=True)
    X_test_tmlp = df_test_tmlp[continuous_features + [f'{cont}_class' for cont in continuous_features]].copy(deep=True)
            
    tmlp_parameters = {        
        'predictors': continuous_features + [f'{cont}_class' for cont in continuous_features],
        'target': target,
        'model': model,
        'model_parameters': {
            'learning_rate': 0.0009,
            'weight_decay': 0.000001,
            'epochs': 150,
            'batch_size': 2 ** 10,
            'reduce_lr_factor': 0.8,
            'reduce_lr_patience': 5,
            'reduce_lr_min': 0.000001,
            'early_stopping_min_delta': 0.0001,
            'early_stopping_patience': 15
        },
        'seeds': [541992]
    }
        
    tmlp_model = NeuralNetworks(**tmlp_parameters)
    tmlp_model.train_and_predict_tmlp(X_train_tmlp, y_train_tmlp, X_test_tmlp)
    
    del df_train_tmlp, df_test_tmlp, X_train_tmlp, y_train_tmlp, X_test_tmlp
    del tmlp_preprocessor, tmlp_parameters, tmlp_model
    
    print('Saving TMLP OOF and Test predictions to current working directory.')
    df_train_processed[['id', 'TMLPPredictions']].to_csv('tmlp_oof_predictions.csv', index=False)
    df_test_processed[['id', 'TMLPPredictions']].to_csv('tmlp_test_predictions.csv', index=False)
    
else:
    print('Loading TMLP OOF and Test predictions from current working directory.')
    df_train_processed['TMLPPredictions'] = pd.read_csv('tmlp_oof_predictions.csv')['MLPPredictions']
    df_test_processed['TMLPPredictions'] = pd.read_csv('tmlp_test_predictions.csv')['MLPPredictions']    
    oof_score = mean_squared_error(df_train_processed['target'], df_train_processed['TMLPPredictions'], squared=False)
    print(f'TMLP OOF RMSE: {oof_score:.6}')
    NeuralNetworks._plot_predictions(None, df_train_processed[target], df_train_processed['TMLPPredictions'], df_test_processed['TMLPPredictions'])


if TRAIN_RMLP:
    model = 'RMLP'
    rmlp_preprocessor = Preprocessor(train=df_train, test=df_test,
                                     n_splits=5, shuffle=True, random_state=cross_validation_seed, scaler=StandardScaler,
                                     create_features=False, discretize_features=False)
    df_train_rmlp, df_test_rmlp = rmlp_preprocessor.transform()
    
    for feature in continuous_features:
        df_train_rmlp[f'{feature}_square'] = df_train_rmlp[feature] ** 2
        df_test_rmlp[f'{feature}_square'] = df_test_rmlp[feature] ** 2
    
    print(f'\n{model} Training Set Shape = {df_train_rmlp.shape}')
    print(f'{model} Training Set Memory Usage = {df_train_rmlp.memory_usage().sum() / 1024 ** 2:.2f} MB')
    print(f'{model} Test Set Shape = {df_test_rmlp.shape}')
    print(f'{model} Test Set Memory Usage = {df_test_rmlp.memory_usage().sum() / 1024 ** 2:.2f} MB\n')

    X_train_rmlp = df_train_rmlp[continuous_features + ['fold'] + [f'{feat}_square' for feat in continuous_features]].copy(deep=True)
    y_train_rmlp = df_train_rmlp[target].copy(deep=True)
    X_test_rmlp = df_test_rmlp[continuous_features + [f'{feat}_square' for feat in continuous_features]].copy(deep=True)
                
    rmlp_parameters = {        
        'predictors': continuous_features + [f'{feat}_square' for feat in continuous_features],
        'target': target,
        'model': model,
        'model_parameters': {
            'learning_rate': 0.01,
            'epochs': 150,
            'batch_size': 2 ** 10,
            'reduce_lr_factor': 0.8,
            'reduce_lr_patience': 5,
            'reduce_lr_min': 0.000001,
            'early_stopping_min_delta': 0.0001,
            'early_stopping_patience': 15
        },
        'seeds': [541992, 721991, 1337, 42, 0]
    }
        
    rmlp_model = NeuralNetworks(**rmlp_parameters)
    rmlp_model.train_and_predict_rmlp(X_train_rmlp, y_train_rmlp, X_test_rmlp)
    
    del df_train_rmlp, df_test_rmlp, X_train_rmlp, y_train_rmlp, X_test_rmlp
    del rmlp_preprocessor, rmlp_parameters, rmlp_model
    
    print('Saving RMLP OOF and Test predictions to current working directory.')
    df_train_processed[['id', 'RMLPPredictions']].to_csv('rmlp_oof_predictions.csv', index=False)
    df_test_processed[['id', 'RMLPPredictions']].to_csv('rmlp_test_predictions.csv', index=False)
    
else:
    print('Loading RMLP OOF and Test predictions from current working directory.')
    df_train_processed['RMLPPredictions'] = pd.read_csv('rmlp_oof_predictions.csv')['RMLPPredictions']
    df_test_processed['RMLPPredictions'] = pd.read_csv('rmlp_test_predictions.csv')['RMLPPredictions']    
    oof_score = mean_squared_error(df_train_processed['target'], df_train_processed['RMLPPredictions'], squared=False)
    print(f'RMLP OOF RMSE: {oof_score:.6}')
    NeuralNetworks._plot_predictions(None, df_train_processed[target], df_train_processed['RMLPPredictions'], df_test_processed['RMLPPredictions'])

prediction_columns = [col for col in df_train_processed.columns if col.endswith('Predictions')]
fig = plt.figure(figsize=(12, 12), dpi=100)
sns.heatmap(df_train_processed[prediction_columns + [target]].corr(),
            annot=True,
            square=True,
            cmap='coolwarm',
            annot_kws={'size': 15},
            fmt='.4f')

plt.tick_params(axis='x', labelsize=18, rotation=90)
plt.tick_params(axis='y', labelsize=18, rotation=0)
plt.title('Prediction Correlations', size=20, pad=20)
plt.show()