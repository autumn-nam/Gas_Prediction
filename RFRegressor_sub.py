# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 09:16:08 2021

@author: user
"""
"""
===============================================================================
라이브러리 로드
===============================================================================
"""
import os
import numpy as np
import pandas as pd 


from sklearn.neural_network import MLPRegressor 
from sklearn.neighbors import KNeighborsRegressor 
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor

from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict 

"""
===============================================================================
데이터 준비 
===============================================================================
"""
os.chdir('../Gas_Prediction')
train = pd.read_csv('./data/total_temp_13_15_pred_df.csv', index_col = 0)
train.index = pd.DatetimeIndex(train.index)



### 공급사별 나누기
A=train.query("구분 == 'A'").drop(['구분'], axis = 1)
B=train.query("구분 == 'B'").drop(['구분'], axis = 1)
C=train.query("구분 == 'C'").drop(['구분'], axis = 1)
D=train.query("구분 == 'D'").drop(['구분'], axis = 1)
E=train.query("구분 == 'E'").drop(['구분'], axis = 1)
G=train.query("구분 == 'G'").drop(['구분'], axis = 1)
H=train.query("구분 == 'H'").drop(['구분'], axis = 1)

### 공급사별 tran test X y 나누기 
X_train_A = A[:'2017'].drop(['공급량'], axis = 1); y_train_A= A[:'2017']['공급량']; X_test_A = A['2019':].drop(['공급량'], axis = 1)
X_train_B = B[:'2017'].drop(['공급량'], axis = 1); y_train_B= B[:'2017']['공급량']; X_test_B = B['2019':].drop(['공급량'], axis = 1)
X_train_C = C[:'2017'].drop(['공급량'], axis = 1); y_train_C= C[:'2017']['공급량']; X_test_C = C['2019':].drop(['공급량'], axis = 1)
X_train_D = D[:'2017'].drop(['공급량'], axis = 1); y_train_D= D[:'2017']['공급량']; X_test_D = D['2019':].drop(['공급량'], axis = 1)
X_train_E = E[:'2017'].drop(['공급량'], axis = 1); y_train_E= E[:'2017']['공급량']; X_test_E = E['2019':].drop(['공급량'], axis = 1)
X_train_G = G[:'2017'].drop(['공급량'], axis = 1); y_train_G= G[:'2017']['공급량']; X_test_G = G['2019':].drop(['공급량'], axis = 1)
X_train_H = H[:'2017'].drop(['공급량'], axis = 1); y_train_H= H[:'2017']['공급량']; X_test_H = H['2019':].drop(['공급량'], axis = 1)


model = RandomForestRegressor()
pred_A = model.fit(X = X_train_A, y = y_train_A).predict(X_test_A)
pred_B = model.fit(X = X_train_B, y = y_train_B).predict(X_test_B)
pred_C = model.fit(X = X_train_C, y = y_train_C).predict(X_test_C)
pred_D = model.fit(X = X_train_D, y = y_train_D).predict(X_test_D)
pred_E = model.fit(X = X_train_E, y = y_train_E).predict(X_test_E)
pred_G = model.fit(X = X_train_G, y = y_train_G).predict(X_test_G)
pred_H = model.fit(X = X_train_H, y = y_train_H).predict(X_test_H)


comp_lst = [train['구분'].unique()]
ans = list()
ans.extend(pred_A)
ans.extend(pred_B)
ans.extend(pred_C)
ans.extend(pred_D)
ans.extend(pred_E)
ans.extend(pred_G)
ans.extend(pred_H)

len(ans)
os.listdir('./data')
te = pd.read_csv('./data/sample_submission.csv')
te['공급량']= ans

te.to_csv("./data/temp_all_13_15_submission.csv", index=False)