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
    
    best_nb = SVC(**study.best_params).fit(
        pd.concat([X_tr, X_val], axis=0),
        pd.concat([y_tr, y_val], axis=0),
        )
    
    return best_nb, study.best_value, study


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
        # 'max_features': trial.suggest_categorical(
        #     'max_features', [None, 'sqrt', 'log2']),
        'max_features': trial.suggest_float('max_features', 0.1, 1),
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
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    
    #%% Overall settings
    run_new_submission = False
    print('\nMake new submission file <{}>'.format(
        run_new_submission))
    
    cpu_use = 5 #int(2*cpu_count()/4)

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
        
    #%% (Test) Manual elements replacing for important features
    from difflib import SequenceMatcher
    
    a = dict_tr[2017]['bq4_1a']
    chg_log = []
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
                    chg_log.append((w_i, w_j))
                    s_new.at[i] = w_j
                    # sleep(1)
        return s_new
    
    
    word_changer(a)
    
    chg_log = list(set(chg_log))    
    chg_log.sort(key=lambda x: x[1])
    
    dict_mnl_chg = {
        2017:[
         ('1종대형면허', '1종대형면허증'),
         ('1종대형운전면허자격증', '1종대형면허증'),
         ('1종대형면허자격증', '1종대형먼허증'),
         ('1종대형운전면허증', '1종대형면허증')
         ('1종대형운전면허', '1종대형면허증'),
         ('자동차대형1종면허', '1종대형면허증'),
         ('1종대형자동차면허증', '1종대형면허증'),
         ('자동차대형1종면허', '1종대형면허증')
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
         ('농기계수리정비사자격증', '농기계정비기사')
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
         ('운전면허증1,2종', '1종운전면허증')
         ('운전면허1,2종', '1종운전면허증'),
         ('1종운전면허', '1종운전면허증'),
         ('운전면허증1종', '1종운전면허증'),
         ('운전면허1종', '1종운전면허증')
         ('운전면허증1종', '1종운전면허증'),
         ('원동기면허', '원동기면허증'),
         ('유리시공기능사', '유리시공기사'),
         ('유치원2급정교사자격증', '유치원교사자격증',)
         ('유치원정교사자격증', '유치원교사자격증',),
         ('유치원교사', '유치원교사자격증')
         ('유치원정교사', '유치원교사자격증',),
         ('육군3사관학교', '육군사관학교'),
         ('국제의료관광코디네이터', '의료관광코디네이터'),
         ('의료보조기사', '의료보조기기사'),
         ('의지보조기기기사자격증', '의료보조기기사')
         ('의지보조기기사자격증', '의료보조기기사'),
         ('의지보조기기자격증', '의료보조기기사'), 
         ('의지보조기기사자격증', '의료보조기기사'),
         ('의지보조기기사', '의지보조기사'),
         ('이미용사자격증', '미용사자격증')
         ('이미용자격증', '미용사자격증'),
         ('이용사자격증','미용사자격증'),
         ('일식조리자격증', '일식조리사자격증'),
         ('임상병리사면허', '임상병리사면허증'),
         ('자동차운전면허', '1종운전면허증')
         ('자동차운전면허증', '1종운전면허증'),
         ('자동차정비기능사', '자동차정비기사'),
         ('자동차정비', '자동차정비기사'),
         ('자동차정비사', '자동차정비기사'),
         ('자동차정비강사', '자동차정비기사'),
         # ('자동차정비기사', '자동차정비사'),
         ('자동차정비원자격증', '자동차정비기사')
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
         ('중고등2급정교사', '중등교사자격증') 
         ('중등2급정교사', '중등교사자격증'),
         ('중등교사2급', '중등교사자격증') 
         ('중등2급정교사', '중등교사자격증'),
         ('중등교원자격', '중등교사자격증')
         ('중등교원자격증', '중등교사자격증'),
         ('중등교사2급', '중등교사자격증'),
         ('중등정교사2급', '중등교사자격증'),
         ('중등학교2급정교사', '중등교사자격증')
         ('중등학교정교사(2급)', '중등교사자격증'),
         ('중등학교정교사2급', '중등교사자격증')
         ('중등학교정교사(2급)', '중등교사자격증'),
         ('철도교통관제', '철도교통관제사'),
         ('철도신호기능사', '철도신호기사'),
         ('철도차량운전면허증', '철도차량운전면허'),
         ('초등정교사자격증', '초등교사자격증'),
         ('초등학교1,2급정교사', '초등교사자격증')
         ('초등학교정교사1급', '초등교사자격증'),
         ('초등학교정교사(1급)', '초등교사자격증')
         ('초등학교정교사1급', '초등교사자격증'),
         ('초등학교1,2급정교사', '초등교사자격증')
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
         ('호스피스간호사', '간호사면허증')
         ('호스피스간호', '간호사면허증'),
         ('호텔서비스사', '호텔서비스'),
         ('화물운송자자격증', '화물운송자격증'),
         ('회계자격증', '회계사자격증')
         ]
                    }
    
    #%% (Test) Remove contents in parenthesis
    def rmv_parenthesis(df):
        for col in tqdm(df.columns):
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
    
    #%% Train naive bayes models
    #mdl_nb = {y:GaussianNB().fit(X_tr[y], y_tr[y]) for y in years}
    
    #%% RandomForest hyperparameter search by optuna
    mdl_selc = 'rf'
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
    elif mdl_selc == 'svc':
        rslt_param_opt = \
            {y: get_svc_optuna(X_tr[y], y_tr[y], X_val[y], y_val[y], num_trial)
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
        
        
    
    
    
    
    
    
    
