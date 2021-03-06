# know_job_rcmd
### DACON Job recommend competition by know data
## Overview
- 공공빅데이터 인턴쉽 프로젝트 같은 조였던 김대근, 최성렬, 황산하로 한 팀(Team153)을 이뤄 도전했던 데이콘 대회
- 결과는 38등/350팀
- 인턴 경험을 통해 초반에 업무 기획부터 결과 제출까지 기획하고 진행
- 주어진 데이터가 년도별로 구성되어있어 `딕셔너리`구조로 데이터를 처리함
- 피쳐의 개수가 많은 데이터에는 어떻게 처리를 해야 모델의 성능이 좋아질지 많은 고민이 되었던 대회

## ROLE
- `김대근` : 데이터 모델링, 데이터 전처리, 파라미터 튜닝
- `황산하` : 분석 기획, 데이터 전처리, 파라미터 튜닝
- `최성렬` : 분석 기획, 데이터 전처리

## Model
- sklearn random forest classifier
  * 주력 모델
  * 
- sklearn SVC
- sklearn SVC + random forest
- sklearn Gaussian Naive Bayes
  * 가장 처음 시도한 모델
  * 
- sklearn Multinomial Naive Bayes
- sklearn Logistic Regression
- catboost
- xgboost random forest classifier
- xgboost classifier

## Hyperparameter tuning
- Using optuna
  * Bayesian hyperparameter optimization framework
- Train, validation split = 0.8 : 0.2 (*참고 shuffle=True, stratify=True) 
- Random state fixed as 42
- Hyperparameter 최적화 목표 : Maximize validation macro f1 score

## Data preprocessing
- 결측치는 -1로 대체함
- Sklearn LabelEncoder를 이용해 feature labeling
- If an element of categorical feature in test set did not exist in training set then replace it as -2

## FeedBack
### 잘한점
1. 분석기획을 하고 진행함으로써 체계적인 프로젝트를 진행할 수 있었음
2. 역할 분배가 적절히 되어서 여유롭게 여러모델을 비교 및 적용할 수 있었음
3. 딕셔너리 구조로 데이터를 조작함으로써 보다 더 깔끔하고 빠르게 데이터를 전처리할 수 있었음
4. 주관식 문항은 같은 의미라도 다르게 표현 하는 경우가 많았기 때문에 의미에 따라 데이터를 처리했음

### 못한점(아쉬운점)
1. 설문지 데이터로 이뤄져 있어 년도별로 설문 문항이 달랐는데 이를 하나로 합치려는 기획을 했다가 실패함 -> 년도별로 학습
2. 성능이 어느 한 시점에서 늘어나지 않았는데 차원의 저주를 풀지 못한것으로 보임( 피쳐의 개수가 각각 약 150이상)
3. 논문을 뒤늦게 찾아보고 적용시켜보려했을 땐 시간이 많이 부족했었음, 학습시간이 꽤 길었기 때문에 실험적인 시도를 마지막에는 많이 하지 못했음
4. `IDEA` 적절하게 자연어 처리를 거쳐서 문항이 달라도 하나의 피쳐로 포함될 수 있는 코드를 만든다면 피쳐도 줄일 수 있고 년도별로 문항이 달라도 공통된 모델에 적용 할 수 있지 않을까 하는 생각이 들었음
5. 딥러닝을 적용해보고 싶었는데 신경망 구축에 실패함.. shape이 안맞았던 것이 원인
6. 차원이 컸기 때문에 Catboost를 GPU환경에서 이용해보고자 코랩을 결제했는데 결과는 생각만큼 좋지 않았음

## FINAL Result
- F1 Score : 0.616
- Optuna를 이용해서 파라미터 튜닝
- 모델은 `RandomForest`가 가장 성능이 뛰어났고 빨랐음
- 자격증에 대한 변수중요도가 높았기 때문에 one-hot encoding을 통해 적절히 학습
![image](https://user-images.githubusercontent.com/57973170/153590671-2ad44928-3dc1-41da-b760-1ef63f8019af.png)

