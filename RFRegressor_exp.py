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

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

model = RandomForestRegressor()
# =============================================================================
# HELPER functions
# =============================================================================

def train_test_split(data, length):
    yr_lst = list(data.index.year.unique())[:-2]
    length = length-1
    
    train = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2018}
    test = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2018}
    
    for yr in yr_lst:
        if yr+length == 2018:
            break
        else:
            pass
        
        train[yr] = data[str(yr):str(yr+length)]
        test[yr] = data[str(yr+length+1):str(yr+length+1)+'-03']
        
        
        
    train_X = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2018}
    train_y = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2018}
    test_X = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2018}
    test_y = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2018}

    for yr in yr_lst:
        if yr+length == 2018:
            break
        else:
            pass
        
        train_X[yr] = train[yr].drop(['공급량'], axis = 1)
        train_y[yr] = train[yr]['공급량']
        test_X[yr] = test[yr].drop(['공급량'], axis = 1)
        test_y[yr] = test[yr]['공급량']
    
        
       
    return train_X, train_y, test_X, test_y


# =============================================================================
# TEST
# tr_X, tr_y, te_X, te_y = train_test_split(A, 5)
# # =============================================================================
def NMAE(true, pred):
    score = np.mean((np.abs(true-pred))/true)
    return score 

#list(te_y.keys())[0]

def get_mean_nmae(tr_X, tr_y, te_X, te_y, model):
    score_lst = []
    yr_lst = list(data.index.year.unique())[:-2]
    
    
    for yr in list(tr_X.keys()):
        pred = model.fit(tr_X[yr], tr_y[yr]).predict(te_X[yr])
        score = NMAE(te_y[yr], pred)
        score_lst.append(score)
        #print(f"{yr}년부터  {score}")
        
    return np.mean(score_lst)

def get_tr_tt_loss(tr_X, tr_y, te_X, te_y, model):
    score_lst = []
    
    yr_lst = list(data.index.year.unique())[:-2]
    
    
    for yr in list(tr_X.keys()):
        sc = pd.Series()
        pred = model.fit(tr_X[yr], tr_y[yr]).predict(te_X[yr])
        fits = model.fit(tr_X[yr], tr_y[yr]).predict(tr_X[yr])
        tr_score = NMAE(tr_y[yr], fits)
        tt_score = NMAE(te_y[yr], pred)
        sc['tr_loss'] = tr_score
        sc['tt_loss']= tt_score
        
        score_lst.append(sc)
        #print(f"{yr}년부터  {sc}")
        
    return score_lst



def runRF(data):
    scr_lst = {comp:np.nan for comp in data['구분'].unique()}
    for comp in list(data['구분'].unique()):
        ## 데이터셋 나누기
        comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
        ## 데이터 별로 nmae 얻기 
        print(f"=======  {comp} 공급사  ========")
        tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
        score = get_tr_tt_loss(tr_X, tr_y, tt_X, tt_y, model)
        scr_lst[comp] = score
    return scr_lst

def imp_plot(importance, comp):
    ## make feature:importance value series
    imp = pd.Series()
    cols = tr_X[2013].columns
    for idx, col in enumerate(cols):
        imp[col] = importance[idx]
        ## sort by values
        imp.sort_values(ascending = False, inplace = True)

    # summarize feature importance
    for i,v in enumerate(imp):
        print(f'Feature: {imp.index[i]}, Score: {v:.3f}')
        # plot feature importance
    sns.barplot(y = imp.index, x = imp)
    plt.title(f"{comp} 공급사")
    plt.show()

    
# =============================================================================
# 예시
# scr_lst = runRF(data)
# test = []
# for comp in scr_lst.keys():
#     print(f'{comp} 공급사')
#     print(f"train loss = {scr_lst[comp][0][0]:.4f}, test_loss = {scr_lst[comp][0][1]:.4f}")
#     test.append(scr_lst[comp][0][1])
# print(f"test_loss mean = {np.mean(test):.4f}")
# 
# =============================================================================
# =============================================================================
# 데이터 준비
# =============================================================================
os.chdir('../Gas_Prediction')
train = pd.read_csv('./data/train_18.csv', index_col = 0)
train.index = pd.DatetimeIndex(train.index)
test = pd.read_csv('./data/test_18.csv', index_col = 0)
test.index = pd.DataFrame(test.index)


# =============================================================================
# 날씨 전부다 사용 (기온, 체감, 지중, +alpha)
#['구분', '공급량', '기온(°C)', '풍속(m/s)', '습도(%)', '증기압(hPa)', '이슬점온도(°C)',
#       '현지기압(hPa)', '해면기압(hPa)', '지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)',
#       '20cm 지중온도(°C)', '30cm 지중온도(°C)', 'day', 'week', 'year', 'vday',
#       '체감온도']
# =============================================================================

data = train
data.columns
scr_lst = runRF(data)
test = []
for comp in scr_lst.keys():
    print(f'{comp} 공급사')
    print(f"train loss = {scr_lst[comp][0][0]:.4f}, test_loss = {scr_lst[comp][0][1]:.4f}")
    test.append(scr_lst[comp][0][1])
print(f"test_loss mean = {np.mean(test):.4f}")

"""
A 공급사
train loss = 0.0164, test_loss = 0.0554
B 공급사
train loss = 0.0200, test_loss = 0.0604
C 공급사
train loss = 0.4321, test_loss = 0.1028
D 공급사
train loss = 0.0199, test_loss = 0.0536
E 공급사
train loss = 0.0178, test_loss = 0.0654
G 공급사
train loss = 0.0150, test_loss = 0.0829
H 공급사
train loss = 0.0216, test_loss = 0.0635
test_loss mean = 0.0691

"""

# =============================================================================
# 온도 관련 전체 다 사용  체감 빼고
# 사용 X = ['구분', '공급량', '기온(°C)', '풍속(m/s)', '습도(%)', '증기압(hPa)', '이슬점온도(°C)',
#       '현지기압(hPa)', '해면기압(hPa)', '지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)',
#       '20cm 지중온도(°C)', '30cm 지중온도(°C)', 'day', 'week', 'year', 'vday']
# =============================================================================

data = train.drop(['체감온도'], axis = 1)
scr_lst = runRF(data)
test = []
for comp in scr_lst.keys():
    print(f'{comp} 공급사')
    print(f"train loss = {scr_lst[comp][0][0]:.4f}, test_loss = {scr_lst[comp][0][1]:.4f}")
    test.append(scr_lst[comp][0][1])
print(f"test_loss mean = {np.mean(test):.4f}")

data train
scr_lst = {comp:np.nan for comp in data['구분'].unique()}
model = RandomForestRegressor()
for comp in list(data['구분'].unique()):
    ## 데이터셋 나누기
    comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
    #comp_data = comp_data[(comp_data.index.month == 1)|(comp_data.index.month == 2)|(comp_data.index.month == 3)]
    #nmae_mean_dct = {length+1:np.nan for length in range(5)}
    ## 데이터 별로 nmae 얻기 
    print(f"=======  {comp} 공급사  ========")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_tr_tt_loss(tr_X, tr_y, tt_X, tt_y, model)
    scr_lst[comp] = score

test = []
for comp in scr_lst.keys():
    print(f'{comp} 공급사')
    print(f"train loss = {scr_lst[comp][0][0]:.4f}, test_loss = {scr_lst[comp][0][1]:.4f}")
    test.append(scr_lst[comp][0][1])
print(f"test_loss mean = {np.mean(test):.4f}")

"""
A 공급사
train loss = 0.0164, test_loss = 0.0556
B 공급사
train loss = 0.0199, test_loss = 0.0609
C 공급사
train loss = 0.4240, test_loss = 0.1027
D 공급사
train loss = 0.0198, test_loss = 0.0533
E 공급사
train loss = 0.0177, test_loss = 0.0653
G 공급사
train loss = 0.0148, test_loss = 0.0831
H 공급사
train loss = 0.0214, test_loss = 0.0642
test_loss mean = 0.0693  **

"""
# =============================================================================
# 온도 관련 4개만 사용 
# 사용 X: ['구분', '공급량', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)', 'day', 'week', 'year', 'vday']
# =============================================================================
data = train.loc[:, ['구분', '공급량', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)', 'day', 'week', 'year', 'vday']]
scr_lst = {comp:np.nan for comp in data['구분'].unique()}
model = RandomForestRegressor()
for comp in list(data['구분'].unique()):
    ## 데이터셋 나누기
    comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
    #comp_data = comp_data[(comp_data.index.month == 1)|(comp_data.index.month == 2)|(comp_data.index.month == 3)]
    #nmae_mean_dct = {length+1:np.nan for length in range(5)}
    ## 데이터 별로 nmae 얻기 
    print(f"=======  {comp} 공급사  ========")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_tr_tt_loss(tr_X, tr_y, tt_X, tt_y, model)
    scr_lst[comp] = score
    

test = []
for comp in scr_lst.keys():
    print(f'{comp} 공급사')
    print(f"train loss = {scr_lst[comp][0][0]:.4f}, test_loss = {scr_lst[comp][0][1]:.4f}")
    test.append(scr_lst[comp][0][1])
print(f"test_loss mean = {np.mean(test):.4f}")


"""
A 공급사
train loss = 0.0177, test_loss = 0.0735
B 공급사
train loss = 0.0215, test_loss = 0.0708
C 공급사
train loss = 0.4649, test_loss = 0.1071
D 공급사
train loss = 0.0211, test_loss = 0.0597
E 공급사
train loss = 0.0191, test_loss = 0.0724
G 공급사
train loss = 0.0162, test_loss = 0.0852
H 공급사
train loss = 0.0225, test_loss = 0.0719
test_loss mean = 0.0772 **
"""
# =============================================================================
# 온도 관련 5개만 사용 (지중온도 + 기온)
# 사용 X: ['구분', '공급량', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)', 'day', 'week', 'year', 'vday']
# =============================================================================
data = train.loc[:, ['구분', '공급량', '기온(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)', 'day', 'week', 'year', 'vday']]
scr_lst = {comp:np.nan for comp in data['구분'].unique()}
model = RandomForestRegressor()
for comp in list(data['구분'].unique()):
    ## 데이터셋 나누기
    comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
    #comp_data = comp_data[(comp_data.index.month == 1)|(comp_data.index.month == 2)|(comp_data.index.month == 3)]
    #nmae_mean_dct = {length+1:np.nan for length in range(5)}
    ## 데이터 별로 nmae 얻기 
    print(f"=======  {comp} 공급사  ========")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_tr_tt_loss(tr_X, tr_y, tt_X, tt_y, model)
    scr_lst[comp] = score
    

test = []
for comp in scr_lst.keys():
    print(f'{comp} 공급사')
    print(f"train loss = {scr_lst[comp][0][0]:.4f}, test_loss = {scr_lst[comp][0][1]:.4f}")
    test.append(scr_lst[comp][0][1])
print(f"test_loss mean = {np.mean(test):.4f}")

"""
A 공급사
train loss = 0.0166, test_loss = 0.0568
B 공급사
train loss = 0.0202, test_loss = 0.0587
C 공급사
train loss = 0.4629, test_loss = 0.1026
D 공급사
train loss = 0.0202, test_loss = 0.0504
E 공급사
train loss = 0.0179, test_loss = 0.0619
G 공급사
train loss = 0.0152, test_loss = 0.0803
H 공급사
train loss = 0.0213, test_loss = 0.0608
test_loss mean = 0.0674  **
"""

# =============================================================================
# 온도 관련 1개만 사용  기온만
# 사용 X: ['구분', '공급량', '기온(°C)', 'day', 'week', 'year', 'vday']
# =============================================================================
data = train.loc[:, ['구분', '공급량', '기온(°C)', 'day', 'week', 'year', 'vday']]
scr_lst = {comp:np.nan for comp in data['구분'].unique()}
model = RandomForestRegressor()
for comp in list(data['구분'].unique()):
    ## 데이터셋 나누기
    comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
    #comp_data = comp_data[(comp_data.index.month == 1)|(comp_data.index.month == 2)|(comp_data.index.month == 3)]
    #nmae_mean_dct = {length+1:np.nan for length in range(5)}
    ## 데이터 별로 nmae 얻기 
    print(f"=======  {comp} 공급사  ========")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_tr_tt_loss(tr_X, tr_y, tt_X, tt_y, model)
    scr_lst[comp] = score
    

test = []
for comp in scr_lst.keys():
    print(f'{comp} 공급사')
    print(f"train loss = {scr_lst[comp][0][0]:.4f}, test_loss = {scr_lst[comp][0][1]:.4f}")
    test.append(scr_lst[comp][0][1])
print(f"test_loss mean = {np.mean(test):.4f}")

"""
A 공급사
train loss = 0.0224, test_loss = 0.0716
B 공급사
train loss = 0.0273, test_loss = 0.0844
C 공급사
train loss = 0.7278, test_loss = 0.1096
D 공급사
train loss = 0.0259, test_loss = 0.0733
E 공급사
train loss = 0.0247, test_loss = 0.0978
G 공급사
train loss = 0.0184, test_loss = 0.0966
H 공급사
train loss = 0.0287, test_loss = 0.0849
test_loss mean = 0.0883
"""




# =============================================================================
# 온도 관련 6개만 사용 (지중온도 + 기온 + 체감)
# 사용 X: ['구분', '공급량', '기온(°C)','5cm 지중온도(°C)', '10cm 지중온도(°C)',
#        '20cm 지중온도(°C)', '30cm 지중온도(°C)', 'day', 'week', 'year', 'vday', '체감온도']
# =============================================================================

data = train.loc[:, ['구분', '공급량', '기온(°C)','5cm 지중온도(°C)', '10cm 지중온도(°C)',
       '20cm 지중온도(°C)', '30cm 지중온도(°C)', 'day', 'week', 'year', 'vday',
       '체감온도']]
scr_lst = {comp:np.nan for comp in data['구분'].unique()}
model = RandomForestRegressor()
for comp in list(data['구분'].unique()):
    ## 데이터셋 나누기
    comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
    #comp_data = comp_data[(comp_data.index.month == 1)|(comp_data.index.month == 2)|(comp_data.index.month == 3)]
    #nmae_mean_dct = {length+1:np.nan for length in range(5)}
    ## 데이터 별로 nmae 얻기 
    print(f"=======  {comp} 공급사  ========")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_tr_tt_loss(tr_X, tr_y, tt_X, tt_y, model)
    
    importance = model.feature_importances_
    imp_plot(importance, comp)    
    
    scr_lst[comp] = score
    
test = []
for comp in scr_lst.keys():
    print(f'{comp} 공급사')
    print(f"train loss = {scr_lst[comp][0][0]:.4f}, test_loss = {scr_lst[comp][0][1]:.4f}")
    test.append(scr_lst[comp][0][1])
print(f"test_loss mean = {np.mean(test):.4f}")


"""
A 공급사
train loss = 0.0168, test_loss = 0.0567
B 공급사
train loss = 0.0204, test_loss = 0.0586
C 공급사
train loss = 0.4731, test_loss = 0.1020
D 공급사
train loss = 0.0204, test_loss = 0.0505
E 공급사
train loss = 0.0181, test_loss = 0.0630
G 공급사
train loss = 0.0154, test_loss = 0.0807
H 공급사
train loss = 0.0214, test_loss = 0.0600
test_loss mean = 0.0673
"""




"""
===============================================================================
제출용
===============================================================================
"""


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

te.to_csv("./data/제출용.csv", index=False)