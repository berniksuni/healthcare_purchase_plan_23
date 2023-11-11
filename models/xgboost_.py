import pandas as pd
# xgboost
from xgboost import XGBRegressor as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
# kfold
from sklearn.model_selection import KFold
# gridsearch
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
# matplotlib
import matplotlib.pyplot as plt
# datetime
import datetime as dt
# numpy
import numpy as np
import sys
sys.path.insert(1, 'C:/Users/berna/Desktop/compes/DATATHON UPC FME 2023/healthcare_challenge/')
from utils import *

# load train data
df_train = pd.read_csv('data/consumo_no23.csv')
df_test = pd.read_csv('data/consumo_23.csv')
df_train = preprocess_cosumo(df_train)
df_test = preprocess_cosumo(df_test)

# train_test_split
X = df_train.drop(['CANTIDADCOMPRA'], axis = 1)
y = df_train['CANTIDADCOMPRA']
folds = KFold(n_splits=10, shuffle=True, random_state=42)

# timer
def timer(start_time=None):
    if not start_time:
        start_time = dt.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((dt.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

# gridsearch for xgboost
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    # 'colsample_bytree': [0.8, 0.9, 1.0],
    # 'gamma': [0, 0.1, 0.2],
    # 'min_child_weight': [1, 3, 5],
    # 'reg_alpha': [0, 0.1, 0.5],
    # 'reg_lambda': [0, 0.1, 0.5],
    # 'objective': ['reg:squarederror','reg:squaredlogerror','reg:linear']
}


# xgboost
xgb = xgb(nthread=10)
# grid_search = RandomizedSearchCV(estimator=xgb, param_distributions=param_grid, cv=folds, n_jobs=10, verbose=2, scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'))
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=folds, n_jobs=10, verbose=2, scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'))
grid_search.fit(X, y)
print(grid_search.best_params_)












