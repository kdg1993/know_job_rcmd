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
    
    from matplotlib import pyplot as plt
    from copy import deepcopy
    from datetime import datetime
    from tqdm import tqdm
    from glob import glob
    from sklearn.naive_bayes import GaussianNB, MultinomialNB
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    
    #%% Overall settings
    run_new_nb_submission = False
    print('\nMake new submission file based on naive bayes <{}>'.format(
        run_new_nb_submission))

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
    
    #%% Train naive bayes model
    mdl_nb = {y:GaussianNB().fit(X_tr[y], y_tr[y]) for y in years}
        
    #%% Check validation score
    y_pred_val = {y:mdl_nb[y].predict(X_val[y]) for y in years}
            
    val_f1 = {y:f1_score(y_val[y], y_pred_val[y], average='macro')
              for y in years}
    
    
        
        
        
        
    
    
    
    
    
    
    
    
