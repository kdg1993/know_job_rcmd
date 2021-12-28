# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 21:01:21 2021

# DACON Job recommend competition based on 'KNOW' data

Main script

@author: KimDaegun
"""

#%% Import packages

#%% Custom functions
def fix_random_seed(seed=42):
    import random
    import numpy as np
    import os
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        tf.random.set_seed(seed)
    except:
        pass
    
#%% Fix random seed
fix_random_seed()

#%% Main script
if __name__ == '__main__':

    #% Import packages
    import numpy as np
    import pandas as pd
    import os
    import pickle
    import itertools as it
    import optuna
    import joblib
    
    from matplotlib import pyplot as plt
    from copy import deepcopy
    from datetime import datetime
    from tqdm import tqdm
    from glob import glob
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from scipy.stats import hmean
    from optuna import Trial, visualization
    from optuna.samplers import TPESampler    
    from multiprocessing import cpu_count
    
    #%% Overall settings
    run_new_submission = False
    print('\nMake new submission file <{}>'.format(
        run_new_submission))
    
    cpu_use = int(2*cpu_count()/4)

    #%% Load data
    years = [2017, 2018, 2019, 2020]
    
    df_smp_subm = pd.read_csv('./data/sample_submission.csv')

    path_tr = sorted(glob('./data/train/*'))
    path_test = sorted(glob('./data/test/*'))
    
    dict_tr = {y:pd.read_csv(p, low_memory=False) 
               for y, p in zip(years, path_tr)}
    dict_test = {y:pd.read_csv(p, low_memory=False) 
               for y, p in zip(years, path_test)}
    
    #%% Check data index match
    # Check index mismatch between train and test
    for y in years:
        i_tr = set(dict_tr[y].idx)
        i_test = set(dict_test[y].idx)
        
        print('\nIndex mismatch between train and test')
        print(y)
        print('Train - Test : ', i_tr.difference(i_test))
        print('Test - Train : ', i_test.difference(i_tr))

    # Check index mismatch between test and submission
    idx_test = pd.concat([dict_test[k].idx for k in dict_test.keys()], axis=0)
    print('\nIndex mismatch between test and submission')
    print('Test - submission : ', set(idx_test) - set(df_smp_subm.idx))
    print('submission - Test : ', set(df_smp_subm.idx) - set(idx_test))  
    
    #%% Check data column match
    # Check column mismatch between train and test
    for year in years:
        c_tr = set(dict_tr[year].columns)
        c_test = set(dict_test[year].columns)
        
        print('\nColumn mismatch between train and test')
        print(year)
        print('Symmetric difference : ', c_tr.symmetric_difference(c_test))
    
    #%% Data preprocessing
    # Set index
    dict_tr = {k:v.set_index('idx') for k, v in dict_tr.items()}
    dict_test = {k:v.set_index('idx') for k, v in dict_test.items()}    
    
    # Fill empty elements
    dict_tr = {k:v.replace(' ', -1) for k, v in dict_tr.items()}
    dict_test = {k:v.replace(' ', -1) for k, v in dict_test.items()}
    
    #%% Make label encoder for each years of datasets
    # Train set label encoding
    dict_encoder = {}
    
    for y, df in dict_tr.items():
        encoder_pack = {}
        
        for col in df.columns:
            try:
                df[col] = df[col].map(float)
                df[col] = df[col].map(int)
            except:
                encoder = LabelEncoder()
                df[col] = df[col].map(str)
                df[col] = encoder.fit_transform(df[col])
                encoder_pack[col] = encoder
                
        dict_encoder[y] = encoder_pack
                
        
    # Test set label encoding
    for y, df in dict_test.items():
        encoder_pack = dict_encoder[y]        
        
        for col in df.columns:
            try:
                df[col] = df[col].map(float)
                df[col] = df[col].map(int)
            except:
                try:
                    encoder = encoder_pack[col]
                    df[col] = df[col].map(str)
                    category_map = {category: idx for idx, category in
                                    enumerate(encoder.classes_)}
                    df[col] = df[col].apply(
                        lambda x: category_map[x] if x in category_map else -2)
                    # -2 indicates unseen in train set
                except:
                    print('\nThere is no encoder for test set', y, col)
                    df[col] = df[col].apply(
                        lambda x: -3 if len(x)>=2 else x)
                    
    #%% (Test) Remove every feature with string value
    '''
    dict_tr = {k:v.drop(columns=dict_encoder[k].keys()) 
               for k, v in dict_tr.items()}
    
    dict_test = {k:v.drop(columns=dict_encoder[k].keys()) 
                 for k, v in dict_test.items()}
    '''
          
    #%% Add values to the dataset to make it positive
    # It is necessary for Multinomial Naive Bayes model
    '''
    min_val_data = 0
    
    for df in list_tr+list_test:
        min_val_data = min(min_val_data, df.min().min())
        
    for df in list_tr+list_test:
        for col in df.columns:
            try:
                df[col] -= min_val_data
            except TypeError:
                pass
    '''
    
    #%% Train validation split (by strati)
    X_tr = {}
    y_tr = {}
    X_val = {}
    y_val = {}
    
    for y, df in dict_tr.items():
        tr, val = train_test_split(df, test_size=0.2, random_state=42,
                                   shuffle=True, stratify=df.knowcode)
        X_tr[y] = tr.drop(columns='knowcode')
        y_tr[y] = tr.knowcode
        X_val[y] = val.drop(columns='knowcode')
        y_val[y] = val.knowcode
    
    #%% Train naive bayes models
    mdl_nb = {y:GaussianNB().fit(X_tr[y], y_tr[y]) for y in years}
    
    #%% Train randomforest models
    '''
    rf_params = {'random_state':42,
                 'n_jobs':-1}
    
    mdl_rf = {y:RandomForestClassifier(**rf_params).fit(X_tr[y], y_tr[y])
              for y in years}
    
    #%% Select model
    mdl = mdl_rf    
    
    #%% Check validation score
    y_pred_val = {y:mdl[y].predict(X_val[y]) for y in years}
            
    dict_val_f1 = {y:f1_score(y_val[y], y_pred_val[y], average='macro')
              for y in years}
    
    #%%
    val_f1_hmean = hmean([v for k, v in dict_val_f1.items()])
    '''
    #%% (Test) optuna
    def objectiveRF(trial: Trial, X, y, X_val, y_val):
        params = {
            'n_estimators': trial.suggest_categorical('n_estimators', [200]),
            'criterion':trial.suggest_categorical('criterion', ['entropy']),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'max_features': trial.suggest_categorical(
                'max_features', [None, 'sqrt', 'log2']),
            'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
            'random_state': trial.suggest_categorical('random_state', [42]),
            }
        
        model = RandomForestClassifier(**params)
        rf_model = model.fit(X, y)
        
        score = f1_score(y_val, rf_model.predict(X_val), average='macro')
        
        return score
    
    
    def get_rf_optuna(X_tr, y_tr, X_val, y_val, n_trial):
        study = optuna.create_study(
            study_name='rf_param_opt',
            direction='maximize', 
            sampler=TPESampler(seed=42)
            )
        
        study.optimize(lambda trial: objectiveRF(
            trial, X_tr, y_tr, X_val, y_val),
            n_trials=n_trial)
        
        best_rf = RandomForestClassifier(**study.best_params).fit(
            pd.concat([X_tr, X_val], axis=0),
            pd.concat([y_tr, y_val], axis=0),
            )
        
        return best_rf, study.best_value, study
        
    rslt_param_opt = \
        {y: get_rf_optuna(X_tr[y], y_tr[y], X_val[y], y_val[y], 15)
         for y in years}
    
    #%% Divide parameter optimization results
    mdl_best = {k:v[0] for k, v in rslt_param_opt.items()}
    dict_val_f1 = {k:v[1] for k, v in rslt_param_opt.items()}
    rslt_opt = {k:v[2] for k, v in rslt_param_opt.items()}
    
    for k, v in dict_val_f1.items():
        print('Val score of {}: {}'.format(k, round(v, 3)))
    
    print('Val harmonic mean score : {}'.format(round(
        hmean([v for k, v in dict_val_f1.items()]), 3)))
    
    #%% Predict test set
    y_pred = [pd.Series(index=dict_test[y].index,
                        data=mdl_best[y].predict(dict_test[y]),
                        name='knowcode')
              for y in years]
    y_pred = pd.concat(y_pred, axis=0)
    
    df_subm = y_pred.reset_index()
        
    #%% Make and Save submission
    if run_new_submission:
        if not os.path.exists('submission_save'):
            os.makedirs('submission_save')
            
        dir_org = os.getcwd()            
        os.chdir('submission_save')
        cur_t = '{}_{}_{}_{}_{}'.format(datetime.now().year, 
                                        datetime.now().month,
                                        datetime.now().day, 
                                        datetime.now().hour,
                                        datetime.now().minute)
        os.mkdir('submission_'+cur_t)
        os.chdir('submission_'+cur_t)
               
        df_subm.to_csv('submission_time_'+cur_t+'.csv',
                       index=False)
        
        try:
            with open ('best_mdl', 'wb') as f:
                pickle.dump(mdl_best, f)
        except MemoryError:
            os.remove('best_mdl')
            print('Pickle Memory error >> Save model by joblib instead')
            joblib.dump(mdl_best, 'best_mdl.pkl')
            
        with open ('param_opt_rslt', 'wb') as f:
            pickle.dump(rslt_opt, f)
        
        os.chdir(dir_org)
        
        
    
    
    
    
    
    
    
