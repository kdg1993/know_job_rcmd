# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 21:01:21 2021

# DACON Job recommend competition based on 'KNOW' data

Main script

@author: KimDaegun
"""

#%% Import packages
from optuna import Trial

#%% Custom functions
def objectiveXGBRF(trial: Trial, X, y, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [200]),
        'objective':trial.suggest_categorical('objective', ['multi:softmax']),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        # 'subsample': trial.suggest_float('subsample', 0.4, 1),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.2, 1),
        # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1),
        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }
    
    model = XGBRFClassifier(**params)
    xgbrf_model = model.fit(X, y)
    
    score = f1_score(y_val, xgbrf_model.predict(X_val), average='macro')

    return score


def get_xgbrf_optuna(X_tr, y_tr, X_val, y_val, n_trial):
    study = optuna.create_study(
        study_name='nb_param_opt',
        direction='maximize', 
        sampler=TPESampler(seed=42)
        )
    
    study.optimize(lambda trial: objectiveXGBRF(
        trial, X_tr, y_tr, X_val, y_val),
        n_trials=n_trial)
    
    best_xgbrf = XGBRFClassifier(**study.best_params).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_xgbrf, study.best_value, study


def objectiveLR(trial: Trial, X, y, X_val, y_val):
    params = {
        'penalty': trial.suggest_categorical('penalty',
                                             ['l1', 'l2', 'elasticnet']),
        'solver': trial.suggest_categorical('solver',
                                             ['saga']),
        'C': trial.suggest_loguniform('C', 1e-3, 1e+3),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }
    
    model = LogisticRegression(**params)
    lr_model = model.fit(X, y)
    
    score = f1_score(y_val, lr_model.predict(X_val), average='macro')

    return score


def get_lr_optuna(X_tr, y_tr, X_val, y_val, n_trial):
    study = optuna.create_study(
        study_name='nb_param_opt',
        direction='maximize', 
        sampler=TPESampler(seed=42)
        )
    
    study.optimize(lambda trial: objectiveLR(
        trial, X_tr, y_tr, X_val, y_val),
        n_trials=n_trial)
    
    best_lr = LogisticRegression(**study.best_params).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_lr, study.best_value, study


def objectiveSVC(trial: Trial, X, y, X_val, y_val):
    params = {
        'kernel': trial.suggest_categorical('kernel', ['rbf', 'sigmoid']),
        'C': trial.suggest_loguniform('C', 1e-3, 1e+3),
        'gamma': trial.suggest_categorical('gamma', ['scale', 'auto'])
        }
    
    model = SVC(**params)
    svc_model = model.fit(X, y)
    
    score = f1_score(y_val, svc_model.predict(X_val), average='macro')

    return score


def get_svc_optuna(X_tr, y_tr, X_val, y_val, n_trial):
    study = optuna.create_study(
        study_name='nb_param_opt',
        direction='maximize', 
        sampler=TPESampler(seed=42)
        )
    
    study.optimize(lambda trial: objectiveSVC(
        trial, X_tr, y_tr, X_val, y_val),
        n_trials=n_trial)
    
    best_svc = SVC(**study.best_params).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_svc, study.best_value, study


def objectiveXGB(trial: Trial, X, y, X_val, y_val):
    params = {
        'objective': trial.suggest_categorical('objective', ['multi:softmax']),
        'n_estimators': trial.suggest_int('n_estimators', 1, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 100),
        # 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1),
        # 'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.1, 1),
        # 'colsample_bynode': trial.suggest_float('colsample_bynode', 0.1, 1),
        'n_jobs': trial.suggest_categorical('n_jobs', [cpu_use]),
        'random_state': trial.suggest_categorical('random_state', [42]),
        }
    
    model = XGBClassifier(**params)
    xgb_model = model.fit(X, y)
    
    score = f1_score(y_val, xgb_model.predict(X_val), average='macro')

    return score


def get_xgb_optuna(X_tr, y_tr, X_val, y_val, n_trial):
    study = optuna.create_study(
        study_name='xgb_param_opt',
        direction='maximize', 
        sampler=TPESampler(seed=42)
        )
    
    study.optimize(lambda trial: objectiveXGB(
        trial, X_tr, y_tr, X_val, y_val),
        n_trials=n_trial)
    
    best_xgb = XGBClassifier(**study.best_params).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_xgb, study.best_value, study


def objectiveRF(trial: Trial, X, y, X_val, y_val):
    params = {
        'n_estimators': trial.suggest_categorical('n_estimators', [200]),
        'criterion':trial.suggest_categorical('criterion', ['entropy']),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'max_features': trial.suggest_float('max_features', 0.1, 1),
        # 'max_samples': trial.suggest_float('max_samples', 0.5, 1),
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
#%%
def objectiveCat(trial: Trial, X, y, X_val, y_val):
    
    param = {
      'random_state': trial.suggest_categorical('random_state', [42]),
      'learning_rate' : trial.suggest_loguniform('learning_rate', 0.01, 0.3),
      'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 0.01, 100.00),
      "n_estimators":trial.suggest_categorical("n_estimators", [1000]),
      "max_depth":trial.suggest_int("max_depth", 4, 16),
      'random_strength' :trial.suggest_int('random_strength', 0, 100),
      "colsample_bylevel":trial.suggest_float("colsample_bylevel", 0.4, 1.0),
      "l2_leaf_reg":trial.suggest_float("l2_leaf_reg",1e-8,3e-5),
      "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
      "max_bin": trial.suggest_int("max_bin", 200, 500),
      'od_type': trial.suggest_categorical('od_type', ['IncToDec', 'Iter'])
  }

    model = CatBoostClassifier(**param)
    cat_model = model.fit(X, y, eval_set=[(X_val, y_val)], verbose=400, early_stopping_rounds=100)
     
    score = f1_score(y_val, cat_model.predict(X_val), average='macro')
    
    return score

def get_cat_optuna(X_tr, y_tr, X_val, y_val, n_trial):
    study = optuna.create_study(
        study_name='cat_param_opt',
        direction='maximize', 
        sampler=TPESampler(seed=42)
        )
    
    study.optimize(lambda trial: objectiveCat(
        trial, X_tr, y_tr, X_val, y_val),
        n_trials=n_trial,
        timeout = 600)
    
    best_cat = CatboostClassifier(**study.best_params).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        
        )
    
    return best_cat, study.best_value, study

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
    import re
    
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
    from optuna import visualization
    from optuna.samplers import TPESampler    
    from multiprocessing import cpu_count
    from xgboost import XGBClassifier, XGBRFClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from catboost import CatBoostClassifier
    from sklearn.metrics import accuracy_score
    
    #%% Overall settings
    run_new_submission = False
    print('\nMake new submission file <{}>'.format(
        run_new_submission))
    
    cpu_use = int(3*cpu_count()/4)

    #%% Load data
    years = [2017, 2018, 2019, 2020]
    
    df_smp_subm = pd.read_csv('G:/dacon/know/sample_submission.csv')

    path_tr = sorted(glob('G:/dacon/know/train/*'))
    path_test = sorted(glob('G:/dacon/know/test/*'))
    
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
    dict_tr = {k:v.replace(' ', '-1') for k, v in dict_tr.items()}
    dict_test = {k:v.replace(' ', '-1') for k, v in dict_test.items()}
    
    # Change elements
    list_chg = (
        ['없음', '없다'], # Integrate '없다' and '없음'
        )
    for pre, post in list_chg:
        dict_tr = {k:v.replace(pre, post) for k, v in dict_tr.items()}
        dict_test = {k:v.replace(pre, post) for k, v in dict_test.items()}

    # Remove space
    for k, v in dict_tr.items():
        for col in v.columns:
            try: 
                v[col].map(float)
            except:
                if not sum(v[col].str.contains(' ')):
                    pass
                else:
                    v[col] = v[col].str.replace(' ', '')
        dict_tr[k] = v
        
    for k, v in dict_test.items():
        for col in v.columns:
            try: 
                v[col].map(float)
            except:
                if not sum(v[col].str.contains(' ')):
                    pass
                else:
                    v[col] = v[col].str.replace(' ', '')
        dict_test[k] = v
        
    # Preprocess the columns bq31, bq30 for 2017, 2018 respectively
    def process_tool_col(y, df):
        if y == 2017:
            s = df.bq31
            # Make all English letters to uppercase letters
            s = s.str.upper() 
            # Change - into ,
            s = s.str.replace('-', ',')
            # Change . into ,
            s = s.str.replace('.', ',')
            # Remove '등'
            s = s.str.replace('등', '')
            df.bq31 = s
        elif y == 2018:
            s = df.bq30
            # Make all English letters to uppercase letters
            s = s.str.upper() 
            # Change - into ,
            s = s.str.replace('-', ',')
            # Change . into ,
            s = s.str.replace('.', ',')
            # Remove '등'
            s = s.str.replace('등', '')
            df.bq30 = s
        else:
            pass # There is no column of tools for job in 2019, 2020
        
        return df
    
    
    dict_tr = {k: process_tool_col(k, v) for k, v in dict_tr.items()}
    dict_test = {k: process_tool_col(k, v) for k, v in dict_test.items()}
    
    #%% Manual elements replacing for important features
    from difflib import SequenceMatcher
    
    a = dict_tr[2018]['bq4_1a']
    chg_log = []
    def word_changer(s):
        s_new = deepcopy(s)
        for i, w_i in s.iteritems():
            for _, w_j in s_new.loc[:i].iteritems():
                sim = SequenceMatcher(None, w_i, w_j).quick_ratio()
                if sim == 1:
                    break
                elif (sim >= 0.8) and (sim < 1):
                    if np.random.random() > 0.99: 
                        print(w_i, ' >> ', w_j, round(sim, 2))
                    chg_log.append((w_i, w_j))
                    s_new.at[i] = w_j
                    # sleep(1)
        return s_new
    
    
    # word_changer(a)
    # chg_log = list(set(chg_log))    
    # chg_log.sort(key=lambda x: x[1])
    
    dict_mnl_chg = {
        2017:[
         ('1종대형면허', '1종대형면허증'),
         ('1종대형운전면허자격증', '1종대형면허증'),
         ('1종대형면허자격증', '1종대형먼허증'),
         ('1종대형운전면허증', '1종대형면허증'),
         ('1종대형운전면허', '1종대형면허증'),
         ('자동차대형1종면허', '1종대형면허증'),
         ('1종대형자동차면허증', '1종대형면허증'),
         ('자동차대형1종면허', '1종대형면허증'),
         ('1종대형자동차운전면허', '1종대형면허증'),
         ('자동차운전면허1종', '1종운전면허증'),
         ('1종보통운전면허증', '1종운전면허증'),
         ('1종보통운전면허', '1종운전면허증'),
         ('1종운전면허', '1종운전면허증'),
         ('운전면허1종', '1종운전면허증'),
         ('3D에니메이터자격증', '3D에니메이션자격증'),
         ('간호사면허', '간호사면허증'),
         ('건설기계면허', '건설기계면허증'),
         ('건설기계기관정비기능사', '건설기계정비기능사'),
         ('건설기계정비기사', '건설기계정비기능사'),
         ('건설기계차체정비기능사', '건설기계정비기능사'),
         ('건설기계정비', '건설기계정비사'),
         ('건설기계정비기사', '건설기계정비사'),
         ('건설재료시험기능사', '건설재료시험기사'),
         ('건설중기정비사', '건설중기정비'),
         ('건축기사자격증', '건축사자격증'),
         ('공조냉동기계기능사', '공조냉동기계기사'),
         ('냉동공조기계기사', '공조냉동기계기사'),
         ('관광통역안내', '관광통역안내사'),
         ('관광통역안내원', '관광통역안내사'),
         ('광산보안기능사', '광산보안기사'),
         ('정교사자격증', '교사자격증'),
         ('교정적5급공무원자격증', '교정직공무원5급자격증'),
         ('한의사국가자격증', '한의사면허증'), 
         ('한의사자격증', '한의사면허증'),
         ('국가자격증(정신보건사회복지사자격증)', '국가자격증(정신보건사회복지사)'),
         ('국내여행안내자격증', '국내여행안내사자격증'),
         ('국제의료코디네이터', '의료관광코디네이터'),
         ('국제의료관광코디네이터', '의료관광코디네이터'),
         ('귀금속가공기능사', '귀금속가공기사'),
         ('그래픽운용기능사', '그래픽운용기사'),
         ('그래픽운용사', '그래픽운용기사'),
         # ('귀금속가공기능사', '금속가공기능사'),
         ('기능사자격증', '기사자격증'),
         ('기록물관리', '기록물관리사'),
         ('기중기운전면허', '기중기운전면허증'),
         ('냉동기계산업기사증', '냉동기계산업기사'),
         ('농기계수리정비사자격증', '농기계정비기사'),
         ('농기계수리정비자격증', '농기계정비기사'),
         # ('기계정비기능사', '농기계정비기능사'),
         ('대기환경기술사', '대기환경기사'),
         ('도배기능사자격증', '도배기사'),
         ('도배기사자격증', '도배기사'),
         ('로더운전사', '로더운전기사'),
         ('무대예술전문', '무대예술전문인'),
         ('무선설비기능사', '무선설비기사'),
         ('무술유단자자격', '무술유단자자격증'),
         ('물리치료면허', '물리치료면허증'),
         ('미용사자격', '미용사자격증'),
         ('이미용사자격증', '미용사자격증'),
         ('방사선사면허', '방사선사면허증'),
         ('방사선전문의', '방사선과전문의면허증'),
         ('방사선과전문의', '방사선과전문의면허증'),
         ('방송통신기능사', '방송통신기사'),
         ('변호사자격', '변호사자격증'),
         ('보일러가스시설시공자격증', '보일러가스시설자격증'),
         ('1종보통운전면허증', '보통1종운전면허'),
         ('사회복지자격증', '사회복지사자격증'),
         ('산업위생관리기술사', '산업위생관리기사'),
         ('상담자격증', '상담사자격증'),
         ('소방설비기술사', '소방설비기사'),
         ('소음진동기사', '소음진동기술사'),
         ('기사자격증', '속기사자격증'),
         ('속기자격증', '속기사자격증'),
         ('의사', '의사면허증'),
         ('의사면허', '의사면허증'),
         ('의사국가면허', '의사면허증'),
         ('수의사국가면허', '수의사면허증'),
         ('수의사면허', '수의사면허증'),
         ('의사자격증', '의사면허증'), 
         ('수의사자격증', '수의사면허증'),
         ('수질환경기술사', '수질환경기사'),
         ('승강기기사', '승강기기능사'),
         ('승강기기능자격증', '승강기기능자격'),
         ('시각디자인기능사', '시각디자인기사'),
         ('애견미용사자격증', '애견미용자격증'),
         ('에너지관리산업기사', '에너지산업관리사'),
         ('영양사면허증', '영양사면허'),
         ('운전면허증1,2종', '1종운전면허증'),
         ('운전면허1,2종', '1종운전면허증'),
         ('1종운전면허', '1종운전면허증'),
         ('운전면허증1종', '1종운전면허증'),
         ('운전면허1종', '1종운전면허증'),
         ('운전면허증1종', '1종운전면허증'),
         ('원동기면허', '원동기면허증'),
         ('유리시공기능사', '유리시공기사'),
         ('유치원2급정교사자격증', '유치원교사자격증'),
         ('유치원정교사자격증', '유치원교사자격증'),
         ('유치원교사', '유치원교사자격증'),
         ('유치원정교사', '유치원교사자격증',),
         ('육군3사관학교', '육군사관학교'),
         ('국제의료관광코디네이터', '의료관광코디네이터'),
         ('의료보조기사', '의료보조기기사'),
         ('의지보조기기기사자격증', '의료보조기기사'),
         ('의지보조기기사자격증', '의료보조기기사'),
         ('의지보조기기자격증', '의료보조기기사'), 
         ('의지보조기기사자격증', '의료보조기기사'),
         ('의지보조기기사', '의지보조기사'),
         ('이미용사자격증', '미용사자격증'),
         ('이미용자격증', '미용사자격증'),
         ('이용사자격증','미용사자격증'),
         ('일식조리자격증', '일식조리사자격증'),
         ('임상병리사면허', '임상병리사면허증'),
         ('자동차운전면허', '1종운전면허증'),
         ('자동차운전면허증', '1종운전면허증'),
         ('자동차정비기능사', '자동차정비기사'),
         ('자동차정비', '자동차정비기사'),
         ('자동차정비사', '자동차정비기사'),
         ('자동차정비강사', '자동차정비기사'),
         # ('자동차정비기사', '자동차정비사'),
         ('자동차정비원자격증', '자동차정비기사'),
         ('자동차정비자격증', '자동차정비기사'),
         ('작업치료사면허', '작업치료사면허증'),
         ('전기기능사자격증', '전기기사'), 
         ('전기기사자격증', '전기기사'),
         ('전문의면허', '전문의면허증'),
         ('전자기기기사', '전자기기기능사'),
         ('전자상거래관리사2급', '전자상거래관리사1급'),
         ('정보처리기능사', '정보처리기사'),
         ('정보처리기술사', '정보처리기사'),
         ('정보처리사', '정보처리기사'),
         ('정보통신기사', '정보통신사'),
         ('정수시설운영관리사', '정수시설운영관리'),
         ('학예사자격증', '정학예사자격증'),
         ('조경기사자격', '조경기사자격증'),
         ('중고등2급정교사', '중등교사자격증'), 
         ('중등2급정교사', '중등교사자격증'),
         ('중등교사2급', '중등교사자격증'), 
         ('중등2급정교사', '중등교사자격증'),
         ('중등교원자격', '중등교사자격증'),
         ('중등교원자격증', '중등교사자격증'),
         ('중등교사2급', '중등교사자격증'),
         ('중등정교사2급', '중등교사자격증'),
         ('중등학교2급정교사', '중등교사자격증'),
         ('중등학교정교사(2급)', '중등교사자격증'),
         ('중등학교정교사2급', '중등교사자격증'),
         ('중등학교정교사(2급)', '중등교사자격증'),
         ('철도교통관제', '철도교통관제사'),
         ('철도신호기능사', '철도신호기사'),
         ('철도차량운전면허증', '철도차량운전면허'),
         ('초등정교사자격증', '초등교사자격증'),
         ('초등학교1,2급정교사', '초등교사자격증'),
         ('초등학교정교사1급', '초등교사자격증'),
         ('초등학교정교사(1급)', '초등교사자격증'),
         ('초등학교정교사1급', '초등교사자격증'),
         ('초등학교1,2급정교사', '초등교사자격증'),
         ('초등학교정교사2급', '초등교사자격증'),
         ('치과의사면허', '치과의사면허증'),
         ('컴퓨터그래픽운용기능사', '컴퓨터,그래픽운용기사'),
         ('컴퓨터그래픽운용기능사', '컴퓨터그래픽스운용기능사'),
         ('컴퓨터그래픽기능사', '컴퓨터그래픽운영기능사'),
         ('컴퓨터그래픽운용기능사', '컴퓨터그래픽운영기능사'),
         ('컴퓨터그래픽운용기능사', '컴퓨터그랙픽운용기능사'),
         ('컴퓨터활용법', '컴퓨터활용'),
         ('특수용접기능사', '특수용접기사'),
         ('폐기물처리기술사', '폐기물처리기사'),
         ('품질관리기술사', '품질관리기사'),
         ('한글속기자격증', '한글속기사자격증'),
         ('의사자격증', '의사면허증'),
         ('의사전문의', '전문의면허증'),
         ('항공기관정비기능', '항공기정비사'),
         ('항공기관정비기능사', '항공기정비사'),
         ('항공정비사면장', '항공기정비사'),
         ('항공정비면장', '항공기정비사'),
         ('항공정비사', '항공기정비사'),
         ('해기사면허증', '해기사면허'),
         ('헬리콥터조종면허증', '헬리콥터조종사면허증'),
         ('호스피스간호사', '간호사면허증'),
         ('호스피스간호', '간호사면허증'),
         ('호텔서비스사', '호텔서비스'),
         ('화물운송자자격증', '화물운송자격증'),
         ('회계자격증', '회계사자격증'),
         ]
                    }
    
    def manual_change(y, df):
        chg_list = dict_mnl_chg[2017]#[y]
        for pre, post in chg_list:
            df = df.replace(pre, post)
        
        return df
       
    dict_tr = {k:manual_change(k, v) for k, v in dict_tr.items()}
    dict_test = {k:manual_change(k, v) for k, v in dict_test.items()}
    
    #%% Remove contents in parenthesis
    def rmv_parenthesis(df):
        for col in df.columns:
            try:
                df[col].map(float)
                pass
            except:
                try:
                    df[col].map(str)
                    df[col] = df[col].replace(r'\([^)]*\)', '', regex=True)
                    # print(col)
                except:
                    pass
                    
        return df
    
    dict_tr = {k: rmv_parenthesis(v) for k, v in dict_tr.items()}
    dict_test = {k: rmv_parenthesis(v) for k, v in dict_test.items()}    
    
    #%% (Test) Make One-hot encoding DataFrame by 'bq31'
    def one_hot_tool_col(df):
        s = df.bq31
        vals = []
        for i, v in s.iteritems():
            vals += v.split(',')
        
        vals = list(set(vals))
        
        df_tool = pd.DataFrame(np.zeros(shape=(len(df.index), len(vals))),
                               index=df.index,
                               columns=vals,
                               dtype='int32')
        
        for i, v in tqdm(s.iteritems(), total=len(df.index)):
            df_tool.loc[i, list(set(v.split(',')))] = 1
            
        return df_tool
    
    df_tool = one_hot_tool_col(dict_tr[2017])
    #%%
    import torch
    from torch.optim import optimizer
    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda': torch.cuda.manual_seed_all(42)
    
        
    #%% (Test) Simple MLP
    import tensorflow as tf
    from tensorflow.keras.models import load_model, Model
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    from tensorflow.keras.layers import Input, Dense
    from tensorflow.keras.optimizers import Adam
    from keras.utils.np_utils import to_categorical
    
    df_tool['knowcode'] = dict_tr[2017].knowcode
    encdr_t_knowcode = LabelEncoder()
    df_tool.knowcode = encdr_t_knowcode.fit_transform(df_tool.knowcode)
    
    one_hot_knowcode = to_categorical(df_tool.knowcode)
    
    X_tool_tr, X_tool_val = train_test_split(df_tool, test_size=0.2,
                                         shuffle=True, 
                                         stratify=df_tool.knowcode)
    
    y_tool_tr = one_hot_knowcode[X_tool_tr.index]
    y_tool_val = one_hot_knowcode[X_tool_val.index]
    
    dir_model_save = './model_save/simple_mlp/'
    if not os.path.exists(dir_model_save):
        os.makedirs(dir_model_save)
        
    es = EarlyStopping(monitor='val_loss', patience=20)
    model_path = dir_model_save + '{epoch:02d}_{val_loss:.5f}.h5'
        
    mc = ModelCheckpoint(filepath=model_path,
                         monitor='val_loss',
                         verbose=0,
                         mode='auto',
                         save_best_only=True)
    
    visible = Input( shape=(X_tool_tr.shape[1],) )
    hidden = Dense(1024, activation='relu')(visible)
    hidden = Dense(1024, activation='relu')(hidden)
    hidden = Dense(1024, activation='relu')(hidden)
    output = Dense(one_hot_knowcode.shape[1], activation='softmax')(hidden)
     
    model = Model(inputs=visible, outputs=output)
    opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    print(model.summary())
                
    model.fit(X_tool_tr, y_tool_tr)
    
    history=model.fit(X_tool_tr, y_tool_tr,
                      validation_data=(X_tool_val, y_tool_val),
                      epochs=100,
                      batch_size=8,
                       callbacks=[mc, es],
                      verbose=1)
    
    raise NotImplementedError
    
    
    #%% (Test) column integration
    '''
    def column_integration(df):
        cols = list(df.columns)
        for c in cols:
            try:
                c1, c2 = c.split('_')
                if re.match('[A-Za-z]+', c1).group() in ['aq', 'kq', 'saq']:
                    neighbor = []
                    for cn in cols:
                        try:
                            cn1, cn2 = cn.split('_')
                            if c1 == cn1 and c2 != cn2:
                                neighbor.append(cn)
                        except:
                            continue
                    
                    new_col = df[c].map(str)
                    for cn in neighbor:
                        new_col += df[cn].map(str)
                    
                    # cols.remove(neighbor)
                    df[c1] = new_col
                    df = df.drop(columns=[c]+neighbor)
                    
            except:
                continue
            
        return df
    
    dict_tr = {k: column_integration(v) for k, v in dict_tr.items()}
    dict_test = {k: column_integration(v) for k, v in dict_test.items()}
    '''
    
    #%% (Test) Word integration by similarity
    '''
    from difflib import SequenceMatcher
    from time import sleep
    
    def word_changer(s):
        s_new = deepcopy(s)
        for i, w_i in s.iteritems():
            for _, w_j in s_new.loc[:i].iteritems():
                sim = SequenceMatcher(None, w_i, w_j).quick_ratio()
                if sim == 1:
                    break
                elif (sim >= 0.9) and (sim < 1):
                    if np.random.random() > 0.99: 
                        print(w_i, ' >> ', w_j, round(sim, 2))
                    chg_log.append((w_i, ' >> ', w_j))
                    s_new.at[i] = w_j
                    # sleep(1)
        return s_new
    
    
    df = deepcopy(dict_tr[2017])
    tot_chg_log = []
    for col in tqdm(df.columns):
        try:
            df[col].map(float)
            continue
        except:
            try:
                print(df[col].head)
                t1 = df[col].map(str)
                
                chg_log = []
                
                t1_new = word_changer(t1)
                chg_log = list(set(chg_log))
                tot_chg_log.append(chg_log)
            except:
                continue
    '''
    
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
                    
    #%% Decrease data size by changing dtype
    dict_tr = {k: v.astype('int32') for k, v in dict_tr.items()}    
    dict_test = {k: v.astype('int32') for k, v in dict_test.items()}
    
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
    for k, v in dict_tr.items():
        dict_tr[k] -= v.min().min()
        dict_test[k] -= v.min().min()
        print(f"Min val of data {k} is {v.min().min()}")
    '''
    
    #%% (Test) Feature importance
    '''
    dir_org = os.getcwd()
    os.chdir(glob(glob('./submission_save/*')[-1])[0])
    pre_study = joblib.load('param_opt_rslt')#('best_mdl.pkl')
    pre_subm = pd.read_csv(glob('*.csv')[0])
    os.chdir(dir_org)
    
    pre_mdl = \
        {y:RandomForestClassifier(**pre_study[y].best_params).fit(
            dict_tr[y].drop(columns='knowcode'),
            dict_tr[y].knowcode) for y in years}
    
    raise NotImplementedError
    
    pre_pred = [pd.Series(index=dict_test[y].index,
                          data=pre_mdl[y].predict(dict_test[y]),
                          name='knowcode')
                for y in years]
    pre_pred = pd.concat(pre_pred, axis=0)
    
    pre_pred = pre_pred.reset_index()
    
    comp_sub_pred = pd.DataFrame(data={
        'subm':pre_subm.knowcode,
        'pred':pre_pred.knowcode,
        'minus':pre_subm.knowcode-pre_pred.knowcode})
    
    raise NotImplementedError
    '''
    
    #%% Train validation split (by stratify)
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
        
    #%% (Test) Make One-hot encoding DataFrame by 'bq31'
    '''
    X_tr_bq31 = df_bq31.loc[X_tr[2017].index, :]   
    X_val_bq31 = df_bq31.loc[X_val[2017].index, :]     
    
    # tune = GaussianNB()
    # tune.fit(X_tr_bq31, y_tr[2017])

    tune = get_svc_optuna(X_tr_bq31, y_tr[2017],
                          X_val_bq31, y_val[2017], 
                          20)
    
    tune = SVC(**{'kernel': 'sigmoid', 
                  # 'C': 24.658329458549105,
                  'gamma': 'scale'})
    tune.fit(X_tr_bq31, y_tr[2017])
    
    score = f1_score(y_val[2017], tune.predict(X_val_bq31), average='macro')
    
    X_tr[2017]['bq31'] = tune.predict(X_tr_bq31)
    X_val[2017]['bq31'] = tune.predict(X_val_bq31)
    '''
    
    #%% Train naive bayes models
    #mdl_nb = {y:GaussianNB().fit(X_tr[y], y_tr[y]) for y in years}
    
    #%% (Test) result of manual change
    '''
    p = {'n_estimators': 200, 'criterion': 'entropy', 'max_depth': 13,
         'max_features': 0.33951195705535725, 'n_jobs': 5, 'random_state': 42}
    m = RandomForestClassifier(**p).fit(X_tr[2017], y_tr[2017])
    fi = pd.DataFrame(data={'nm': X_tr[2017].columns,
                            'sc': m.feature_importances_}).sort_values(
                                by='sc', ascending=False)
    v = deepcopy(X_val[2017])
    v['true'] = y_val[2017]
    v['pred'] = m.predict(X_val[2017])
    vv = v.loc[:, ['bq1', 'bq4_1a', 'bq31', 'true', 'pred']]
    vv['min'] = vv.true-vv.pred
    vv['bq4_1a'] = dict_encoder[2017]['bq4_1a'].inverse_transform(vv['bq4_1a'])
    '''
    #%%
    for y,df in dict_tr.items():
        X_tr[y] = torch.FloatTensor(np.array(X_tr[y])).to(device)
        y_tr[y] = torch.FloatTensor(np.array(y_tr[y])).to(device)
        X_val[y] = torch.FloatTensor(np.array(X_val[y])).to(device)
        y_val[y] = torch.FloatTensor(np.array(y_val[y])).to(device)
    for y,df in dict_test.items():
        dict_test[y] = torch.FloatTensor(np.array(dict_test[y])).to(device)
    
    #%%
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
    from tensorboardX import SummaryWriter
    summary = SummaryWriter()
    
    tr_dataset = {}
    tr_loader = {}
    val_dataset = {}
    val_loader = {}
    
    for y,df in dict_tr.items():
        tr_dataset[y] =TensorDataset(X_tr[y], y_tr[y])
        tr_loader[y] = DataLoader(tr_dataset[y], batch_size=32, shuffle = True,
                                  drop_last=True)
        val_dataset[y] = TensorDataset(X_val[y], y_val[y])
        val_loader[y] = DataLoader(val_dataset[y], batch_size=32, shuffle = True,
                                  drop_last=True)
    #%%
    class DNN(torch.nn.Module):
        def __init__(self):
            super(DNN, self).__init__()
            K = 150
            self.relu = torch.nn.ReLU()
            
            self.Linear1 = torch.nn.Linear(1, 128)
            self.Linear2 = torch.nn.Linear(128, 512)
            self.Linear3 = torch.nn.Linear(512, 1024)
            self.Linear4 = torch.nn.Linear(1024, 2048)
            self.Linear5 = torch.nn.Linear(2048, 1024)
            self.Linear6 = torch.nn.Linear(1024, 512)
            self.Linear7 = torch.nn.Linear(512, 256)
        
        
            self.bn1 = torch.nn.BatchNorm1d(128)
            self.bn2 = torch.nn.BatchNorm1d(512)
            self.bn3 = torch.nn.BatchNorm1d(1024)
            self.bn4 = torch.nn.BatchNorm1d(2048)
            self.bn5 = torch.nn.BatchNorm1d(1024)
            self.bn6 = torch.nn.BatchNorm1d(512)
            
            self.model = torch.nn.Sequential(self.Linear1, self.bn1, self.relu,
                                             self.Linear2, self.bn2, self.relu,
                                             self.Linear3, self.bn3, self.relu,
                                             self.Linear4, self.bn4, self.relu,
                                             self.Linear5, self.bn5, self.relu,
                                             self.Linear6, self.bn6, self.relu,
                                             self.Linear7)
            
            self.weight_init()
            
        def weight_init(self):
            for module in self.model:
                if isinstance(module, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    module.bias.data.fill_(0.01)
        
        def forward(self,x):
            x = x.reshape(-1,1)
            h1 = F.relu(self.Linear1(x))
            h2 = F.relu(self.Linear2(h1))
            h3 = F.relu(self.Linear3(h2))
            h4 = F.relu(self.Linear4(h3))
            h5 = F.relu(self.Linear5(h4))
            h6 = F.relu(self.Linear6(h5))
            h7 = self.model(h6)
            
            return F.log_softmax(h7, dim = 1)
        #%%
        model = DNN().to(device)
        epochs = 10
        batch = 300
        lr = 5e-5
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 1e-6)
        loss_fn = torch.nn.MSELoss().to(device)
        
        for epochs in range(epochs+1):
            avg_loss = 0
            model.train()
            
            for t,df in tr_loader.items():
                for X, y in tqdm(tr_loader[t]):
                    hypothesis = model(X)
                    loss = loss_fn(hypothesis,y)
                    loss = loss**0.5
                    avg_loss += loss/len(tr_loader[t])
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                model.eval()
                with torch.no_grad():
                    avg_val_loss = 0
                    for X_val, y_val in tqdm(val_loader[t]):
                        val_h = model(X_val)
                        loss_val = loss_fn(val_h, y_val)
                        loss_val = loss_val**0.5
                        avg_val_loss += loss_val/len(val_loader[t])
                if epochs % 1 == 0:
                    print("epoch : {:4d}Train_Loss:{:7f}Val_Loss{:.7d}".format(epochs,avg_loss, avg_val_loss))
                    summary.add_scalar('Train_loss', avg_loss.items(), epochs)
                    summary.add_scalar('Validatiom_loss', avg_val_loss.items(), epochs)
            
            
        



            
    #%% RandomForest hyperparameter search by optuna
    mdl_selc = 'cat'
    num_trial = 30
    print('='*15, f'Model selected [{mdl_selc}]', '='*15)
    
    if mdl_selc == 'rf':
        rslt_param_opt = \
            {y: get_rf_optuna(X_tr[y], y_tr[y], X_val[y], y_val[y], num_trial)
             for y in years}
    elif mdl_selc == 'xgb':
        rslt_param_opt = \
            {y: get_xgb_optuna(X_tr[y], y_tr[y], X_val[y], y_val[y], num_trial)
             for y in years}
    elif mdl_selc == 'xgbrf':
        rslt_param_opt = \
            {y: get_xgbrf_optuna(X_tr[y], y_tr[y], X_val[y], y_val[y],
                                 num_trial)
             for y in years}
    elif mdl_selc == 'svc':
        rslt_param_opt = \
            {y: get_svc_optuna(X_tr[y], y_tr[y], X_val[y], y_val[y], num_trial)
             for y in years}
    elif mdl_selc == 'cat':
        rslt_param_opt= \
            {y: get_cat_optuna(X_tr[y], y_tr[y], X_val[y], y_val[y], num_trial)
             for y in years}
            
    #%% Divide parameter optimization results
    mdl_best = {k:v[0] for k, v in rslt_param_opt.items()}
    dict_val_f1 = {k:v[1] for k, v in rslt_param_opt.items()}
    rslt_opt = {k:v[2] for k, v in rslt_param_opt.items()}
    
    for k, v in dict_val_f1.items():
        print('Val score of {}: {}'.format(k, round(v, 3)))
    
    print('Val harmonic mean score : {}'.format(round(
        hmean([v for k, v in dict_val_f1.items()]), 3)))
    
    #%% Feature importance
    f_imp = {y:pd.DataFrame(data={'f_nm': dict_test[y].columns,
                                  'score': mdl_best[y].feature_importances_}).\
             sort_values(by='score', ascending=False)
             for y in years}
    
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
        
        
    
    
    
    
    
    
    
