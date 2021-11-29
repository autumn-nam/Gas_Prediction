"""
X, 컬럼값들 예측 모델 
"""
import os
os.chdir('../Gas_Prediction')
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# =============================================================================
# 데이터 준비 
# Total Data 준비!!!
# =============================================================================

## Weather 데이터
weather = pd.read_csv('./data/기상 원본.csv', encoding = 'ms949')
weather.index = pd.DatetimeIndex(weather['일시'])
weather.info()

## Drop string columns / irrelavant columns
# weather.drop(['지점', '지점명', '일시'], axis = 1, inplace = True)
# weather.drop(list(weather.filter(regex = 'QC')), axis = 1, inplace = True)
weather = weather.loc[:, ['기온(°C)', '풍속(m/s)', '증기압(hPa)', '이슬점온도(°C)', '지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)']]
# weather.to_csv('기상2010_2019.csv')

#weather.columns
### 공급량 데이터 가져오기 
data = pd.read_csv('./data/train.csv', index_col = 0)
data.index = pd.DatetimeIndex(data.index)
test = pd.read_csv('./data/test.csv', index_col = 0)
test.index = pd.DatetimeIndex(test.index)
data = data.append(test)
data.columns
###  공급량 + weather
data[weather.columns] = weather # data + weather

### 주기 데이터 불러오기 
season =  pd.read_csv('./data/total_train.csv', index_col = 0)
season = season[season['구분'] == 'A']
season.index = pd.DatetimeIndex(season.index)
season = season.loc[:, ['day', 'week', 'year', 'vday']]

data[season.columns] = season # data + weather + 주기
###  윤년 제거 
data = data[(data.index.year != 2016) | (data.index.month != 2) | (data.index.day != 29)]
# data.loc['2016-02-28':, :].head(25)

data.columns
## 온도 관련 NA 대체 
for col in ['기온(°C)','풍속(m/s)','5cm 지중온도(°C)', '10cm 지중온도(°C)','지면온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)', '증기압(hPa)', '이슬점온도(°C)' ]:
    data[col+'+Z'] = data[col].fillna(method = 'ffill')
    data[col+'-Z'] = data[col].fillna(method='bfill')
    data[col] = (data[col+'+Z'] + data[col+'-Z']) / 2
    

## 부수적 컬럼 삭제
data = data.drop(list(data.filter(regex = 'Z')), axis = 1)
data = data.drop(list(data.filter(regex = '-Z')), axis = 1)
data.isna().sum()


## 체감온도 생성
data['체감온도'] = 13.127 +(0.6215*data['기온(°C)'])-(13.947*(data['풍속(m/s)']**0.16)) + (0.486*data['기온(°C)']*(data['풍속(m/s)'])**0.16)

### train test 나누기
train = data[:'2018'] 
test = data['2019':]

### 체감온도 생성

### Drop NA
train.isna().sum()
train = train.dropna(axis = 1)
test = test.loc[:, train.columns ]

total_df.columns
## 저장
#total_df = pd.concat([train, test])
#total_df.to_csv('./data/total_df.csv')
#train.to_csv('./data/train_14.csv')
#test.to_csv('./data/test_14.csv')
total_df.columns
total_df.filter(regex = "온도").columns
## 최종 데이터 
dct = {comp :pd.DataFrame() for comp in train['구분'].unique()}
for comp in train['구분'].unique():
    dct[comp] = train.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)

# =============================================================================
# 데이터 시각화 
# =============================================================================
import matplotlib.pyplot as plt
import seaborn as sns

## 컬럼별 Heatmap

### Corr plot
heatmap_plot('A')
heatmap_plot('B')
heatmap_plot('C')
heatmap_plot('D')
heatmap_plot('E')
heatmap_plot('G')
heatmap_plot('H')
# =============================================================================
# Helper Functions
# =============================================================================
import warnings
warnings.filterwarnings('ignore')

import os 
import pandas as pd
import numpy as np
import json
from sklearn.neural_network import MLPRegressor 
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor



#A.index.year.unique()[:-1]
#yr_lst = list(data.index.year.unique())[:-2] # [2013, 2014, 2015, 2016, 2017]
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

def heatmap_plot(dct, comp):
    plt.figure(figsize=(12,10))
    cor = dct[f"{comp}"].corr()
    matrix = np.triu(cor)
    sns.heatmap(cor, annot=True, cmap=plt.cm.Blues, mask = matrix, square = True)
    plt.title(f"{comp}공급사 corr plot", fontsize = 25, fontweight = 'bold')
    plt.show()

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
# RF Importance Plot
# =============================================================================


"""
vday 포함함 

"""
## model fitting
model = RandomForestRegressor()
scr_lst = []
for comp in dct.keys():
    comp_data = dct[comp]
    print(f"공급사 {comp}")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_mean_nmae(tr_X, tr_y, tt_X, tt_y, model)
    
    importance = model.feature_importances_
    imp_plot(importance, comp)
    scr_lst.append(score)
    print(f"{comp} 공급사의 5년단위 스플릿 nmae 평균 = {score}")
    print("==="*10)    
print(f"전체공급사 평균 NMAE = {np.mean(scr_lst)}")

# =============================================================================
# <결과> 
# 공급사 A
# A 공급사의 5년단위 스플릿 nmae 평균 = 0.05561114396809631
# ==============================
# 공급사 B
# B 공급사의 5년단위 스플릿 nmae 평균 = 0.06074821360626744
# ==============================
# 공급사 C
# C 공급사의 5년단위 스플릿 nmae 평균 = 0.1031079506922598
# ==============================
# 공급사 D
# D 공급사의 5년단위 스플릿 nmae 평균 = 0.05326472986944098
# ==============================
# 공급사 E
# E 공급사의 5년단위 스플릿 nmae 평균 = 0.06550779030521467
# ==============================
# 공급사 G
# G 공급사의 5년단위 스플릿 nmae 평균 = 0.08228190624541355
# ==============================
# 공급사 H
# H 공급사의 5년단위 스플릿 nmae 평균 = 0.06424999098669398
# ==============================
# 전체공급사 평균 NMAE = 0.06925310366762667
# =============================================================================

"""
vday 포함 안함
"""
model = RandomForestRegressor()
scr_lst = []
for comp in dct.keys():
    comp_data = dct[comp].drop(['vday'], axis = 1)
    print(f"공급사 {comp}")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)    
    score = get_mean_nmae(tr_X, tr_y, tt_X, tt_y, model)
    importance = model.feature_importances_
    imp_plot(importance, comp)
    
    scr_lst.append(score)
    print(f"{comp} 공급사의 5년단위 스플릿 nmae 평균 = {score}")
    print("==="*10)    
print(f"전체공급사 평균 NMAE = {np.mean(scr_lst)}")

# =============================================================================
# <결과>
# 공급사 A
# A 공급사의 5년단위 스플릿 nmae 평균 = 0.06051718305440067
# ==============================
# 공급사 B
# B 공급사의 5년단위 스플릿 nmae 평균 = 0.06377616614806045
# ==============================
# 공급사 C
# C 공급사의 5년단위 스플릿 nmae 평균 = 0.1035922386015481
# ==============================
# 공급사 D
# D 공급사의 5년단위 스플릿 nmae 평균 = 0.05456972483020012
# ==============================
# 공급사 E
# E 공급사의 5년단위 스플릿 nmae 평균 = 0.06749879779833434
# ==============================
# 공급사 G
# G 공급사의 5년단위 스플릿 nmae 평균 = 0.0922075922744295
# ==============================
# 공급사 H
# H 공급사의 5년단위 스플릿 nmae 평균 = 0.06709844291653218
# ==============================
# 전체공급사 평균 NMAE = 0.07275144937478648
# =============================================================================


# =============================================================================
# 온도만 전부 다 사용 
# =============================================================================

model = RandomForestRegressor()
scr_lst = []

for comp in dct.keys():
    comp_data = dct[comp].loc[:, ['공급량', '기온(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)',
       '20cm 지중온도(°C)', '30cm 지중온도(°C)', 'day', 'week', 'year', 'vday', '체감온도']]
    print(f"공급사 {comp}")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_mean_nmae(tr_X, tr_y, tt_X, tt_y, model)
    
    importance = model.feature_importances_
    imp_plot(importance, comp)
    scr_lst.append(score)
    print(f"{comp} 공급사의 5년단위 스플릿 nmae 평균 = {score}")
    print("==="*10)    
print(f"전체공급사 평균 NMAE = {np.mean(scr_lst)}")
"""

"""
# =============================================================================
# 온도(체감온도, 기온, 지중온도)
# =============================================================================

os.chdir('../Gas_Prediction')
train = pd.read_csv('./data/train_18.csv', index_col = 0)
train.index = pd.DatetimeIndex(train.index)
test = pd.read_csv('./data/test_18.csv', index_col = 0)
test.index = pd.DataFrame(test.index)






# =============================================================================
# 지중온도들 시각화(imp plot 에서 항상 순위권)
# =============================================================================

ground_temp = train[list(train.filter(regex = '지중온도'))]
## lineplot
sns.lineplot(data = ground_temp, x = ground_temp.index) 
fig, ax = plt.subplots(nrows = 4, ncols = 1)
ax = ax.reshape(4)
for idx, col in enumerate(ground_temp.columns):
    sns.lineplot(data = ground_temp, x = ground_temp.index, y = col, ax = ax[idx])

## corr plot
sns.pairplot(ground_temp)
plt.show()

## 
ground_temp.describe()

