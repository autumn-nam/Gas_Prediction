## 기본
import json
import os 
import pandas as pd
import numpy as np
import scipy.stats as sta

import warnings
warnings.filterwarnings('ignore')


## 시각화
import matplotlib.pyplot as plt
import seaborn as sns 

## 전처리 
from sklearn.preprocessing import MinMaxScaler

## 모델링
from sklearn.neural_network import MLPRegressor 
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

plt.rc("figure", figsize=(16, 12))
plt.rc("font", size=13)
# =============================================================================
#  Helper Funcitons
# =============================================================================

def train_test_split(data, length, y_str):
    yr_lst = list(data.index.year.unique())[:-1]
    length = length-1
    
    train = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2019}
    test = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2019}
    
    for yr in yr_lst:
        if yr+length == 2019:
            break
        else:
            pass
        
        train[yr] = data[str(yr):str(yr+length)]
        test[yr] = data[str(yr+length+1):str(yr+length+1)+'-03']
        
        
        
    train_X = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2019}
    train_y = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2019}
    test_X = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2019}
    test_y = {yr:pd.DataFrame() for yr in yr_lst if yr+length < 2019}

    for yr in yr_lst:
        if yr+length == 2019:
            break
        else:
            pass
        
        train_X[yr] = train[yr].drop([y_str], axis = 1)
        train_y[yr] = train[yr][y_str]
        test_X[yr] = test[yr].drop([y_str], axis = 1)
        test_y[yr] = test[yr][y_str]
    
        
       
    return train_X, train_y, test_X, test_y



# =============================================================================
# TEST
# tr_X, tr_y, te_X, te_y = train_test_split()
# # =============================================================================
def NMAE(true, pred):
    score = np.mean((np.abs(true-pred))/true)
    return score 

from sklearn.metrics import mean_squared_error

def get_mean_nmae(tr_X, tr_y, te_X, te_y, model):
    score_lst = []
  
    for yr in list(tr_X.keys()):
        pred = model.fit(tr_X[yr], tr_y[yr]).predict(te_X[yr])
        score = NMAE(te_y[yr], pred)
        score_lst.append(score)
        print(f"{yr}년부터 \n {score}")
        print("-----"*5)
    return np.mean(score_lst)


def get_tr_tt_loss(tr_X, tr_y, te_X, te_y, model):
    score_dct = {}
    for yr in list(tr_X.keys()):
        sc = pd.Series()
        pred = model.fit(tr_X[yr], tr_y[yr]).predict(te_X[yr])
        fits = model.fit(tr_X[yr], tr_y[yr]).predict(tr_X[yr])
        tr_score = NMAE(tr_y[yr], fits)
        tt_score = NMAE(te_y[yr], pred)
        sc['tr_loss'] = tr_score
        sc['tt_loss']= tt_score
        
        score_dct[yr] = sc
        print(f"{yr}년부터")
        print(f"trian_loss = {tr_score:.5f}, test_loss = {tt_score:.5f}")
        print("-----"*10)       
    return score_dct

def get_tr_tt_mse_loss(tr_X, tr_y, te_X, te_y, model):
    score_dct = {}
    for yr in list(tr_X.keys()):
        sc = pd.Series()
        pred = model.fit(tr_X[yr], tr_y[yr]).predict(te_X[yr])
        fits = model.fit(tr_X[yr], tr_y[yr]).predict(tr_X[yr])
        tr_score = mean_squared_error(tr_y[yr], fits)
        tt_score = mean_squared_error(te_y[yr], pred)
        sc['tr_loss'] = tr_score
        sc['tt_loss']= tt_score
        
        score_dct[yr] = sc
        print(f"{yr}년부터")
        print(f"trian_loss = {tr_score:.5f}, test_loss = {tt_score:.5f}")
        print("-----"*10)       
    return score_dct


def get_tr_tt_mse_loss2019(tr_X, tr_y, te_X, te_y, model):
    score_dct = {}        
    targetyear = list(tr_X.keys())[-1]
    for yr in list(tr_X.keys()):
        
        sc = pd.Series()
        pred = model.fit(tr_X[yr], tr_y[yr]).predict(te_X[targetyear])
        fits = model.fit(tr_X[yr], tr_y[yr]).predict(tr_X[yr])
        tr_score = mean_squared_error(tr_y[yr], fits)
        tt_score = mean_squared_error(te_y[targetyear], pred)
        sc['tr_loss'] = tr_score
        sc['tt_loss']= tt_score
        
        score_dct[yr] = sc
        print(f"{yr}년부터, target year = {targetyear}")
        print(f"trian_loss = {tr_score:.5f}, test_loss = {tt_score:.5f}")
        print("-----"*10)       
    return score_dct

def get_tr_tt_mean_loss(score_dct):
    tr=[]; tt=[]
    for key in score_dct.keys():    
        tr.append(score_dct[key][0])
        tt.append(score_dct[key][1])
    return np.mean(tr), np.mean(tt)
    

# =============================================================================
# temp(기온) 예측하기, 데이터 준비
# =============================================================================
os.chdir('../Gas_Prediction')
data= pd.read_csv('./data/total_temps_dec_df.csv', index_col = 0)
data.index = pd.DatetimeIndex(data.index)
data.columns


temp = data[list(data.filter(regex = '기온'))]
temp_gr = data[list(data.filter(regex = '지면'))]
temp5 = data[list(data.filter(regex = '5cm'))]
temp10 = data[list(data.filter(regex = '10'))]
temp20 = data[list(data.filter(regex = '20cm'))]
temp30 = data[list(data.filter(regex = '30cm'))]
temp_f = data[list(data.filter(regex = '체감'))]
temp_hpa = data[list(data.filter(regex = '증기압'))]
temp_drp = data[list(data.filter(regex = '이슬'))]

#### Shift columnn 추가하기
def addShift(data):
    data['shift'] = data[data.columns[0]].shift(1)
    data['shift2'] = data[data.columns[0]].shift(2)
    data.dropna(axis = 0, inplace = True)
    data.loc['2019-01-01 01:00:00':, 'shift'] = np.nan
    data.loc['2019-01-01 01:00:00':, 'shift2'] = np.nan
    return data
def addShift2(data, shift):
    for i in range(shift):
        data['shift'+str(i+1)] = data[data.columns[0]].shift(i+1)
    data.dropna(axis = 0, inplace = True)
    for i in range(shift):
        data.loc['2019-01-01 01:00:00':, 'shift'+str(i+1)] = np.nan

    return data

def addShift3(data):
    data['1시간전'] = data[data.columns[0]].shift(1)
    data['24시간전'] = data[data.columns[0]].shift(24)
    data['1년전'] = data[data.columns[0]].shift(8760)
    data.dropna(axis = 0, inplace = True)
    
    data.loc['2019-01-01 01:00:00':,'1시간전'] = np.nan
    data.loc['2019-01-01 01:00:00':,'24시간전'] = np.nan
    data.loc['2019-01-01 01:00:00':,'1년전'] = np.nan
    
    return data

temp = addShift(temp)
temp_gr = addShift(temp_gr)
temp5 = addShift(temp5)
temp10 = addShift(temp10)
temp20 = addShift(temp20)
temp30 = addShift(temp30)
temp_f = addShift(temp_f)
temp_hpa = addShift(temp_hpa)
temp_drp = addShift(temp_drp)


model = RandomForestRegressor()
tr_X, tr_y, tt_X, tt_y = train_test_split(data = temp5, length = 6, y_str = '5cm 지중온도(°C)')
rf = model.fit(tr_X[2013], tr_y[2013])
ttX = np.array(tt_X[2013].iloc[0, :]).reshape(1, 4)
pred1 = rf.predict(ttX)

_3 = tt_X[0,2]
ttX = np.array(tt_X[2013].iloc[1, :]).reshape(1, 4)
ttX[0][2] = pred
ttX[0][3] = _3

pred = rf.predict(ttX)
ttX = np.array(tt_X[2013].iloc[2, :]).reshape(1, 4) # 다음 시간대 
ttX[0][2] = pred # pred값으로 대체해서 X 완성

tt_X[2013].iloc[2159, :]


model = LGBMRegressor()
temp = addShift2(temp, 1)
ans_lst = []
tr_X, tr_y, tt_X, tt_y = train_test_split(data = temp, length = 6, y_str = '기온(°C)')
rf = model.fit(tr_X[2013], tr_y[2013])
for i in range(tt_X[2013].shape[0]):
    if i == 0:
        ttX = np.array(tt_X[2013].iloc[i, :]).reshape(1, 3)
        pred = rf.predict(ttX)
        ans_lst.extend(pred)
    else:
        ttX = np.array(tt_X[2013].iloc[i, :]).reshape(1, 3) # 다음 시간대 
        ttX[0][2] = pred 
        pred = rf.predict(ttX)
        ans_lst.extend(pred)

ans_sr = pd.Series(ans_lst, index = y_true.index)
y_true = temp.loc['2019':, '기온(°C)']    
mse1 = mean_squared_error(y_true, ans_sr)
mse1
ans_sr.plot();y_true.plot();plt.title(f"mse = {mse1:.4f} with shift1 data", fontsize = 35, fontweight = 'bold')

_= pd.DataFrame(y_true)
_['pred'] = ans_sr
_


temp = addShift2(temp, 2)
ans_lst = []
tr_X, tr_y, tt_X, tt_y = train_test_split(data = temp, length = 6, y_str = '기온(°C)')
rf = model.fit(tr_X[2013], tr_y[2013])


ttX = np.array(tt_X[2013].iloc[0, :]).reshape(1, 4)
_3 = ttX[0,2]
pred = rf.predict(ttX)

ttX = np.array(tt_X[2013].iloc[1, :]).reshape(1, 4)
ttX[0,2] = pred
ttX[0,3] = _3

_3= ttX[0,2] 
pred =rf.predict(ttX)
ttX = np.array(tt_X[2013].iloc[2, :]).reshape(1, 4)
ttX[0,2] = pred
ttX[0,3] = _3

temp = addShift2(temp, 2)
ans_lst = []
tr_X, tr_y, tt_X, tt_y = train_test_split(data = temp, length = 6, y_str = '기온(°C)')
rf = model.fit(tr_X[2013], tr_y[2013])
for i in range(tt_X[2013].shape[0]):
    if i == 0:
        ttX = np.array(tt_X[2013].iloc[i, :]).reshape(1, 4)
        pred = rf.predict(ttX)
        ans_lst.extend(pred)
    else:
        _3 = ttX[0,2]
        ttX = np.array(tt_X[2013].iloc[i, :]).reshape(1, 4) # 다음 시간대 
        ttX[0][2] = pred 
        ttX[0][3] = _3
        pred = rf.predict(ttX)
        ans_lst.extend(pred)
        
print(len(ans_lst)) 
y_true = temp.loc['2019':, '기온(°C)']           
ans_sr = pd.Series(ans_lst, index = y_true.index)
mse2 = mean_squared_error(y_true, ans_sr)
mse2

ans_sr.plot();y_true.plot();plt.title(f"mse = {mse2:.4f} with shift2 data", fontsize = 35, fontweight = 'bold')


temp = addShift2(temp, 2)
ans_lst = []
tr_X, tr_y, tt_X, tt_y = train_test_split(data = temp, length = 6, y_str = '기온(°C)')
rf = model.fit(tr_X[2013], tr_y[2013])

for i in range(tt_X[2013].shape[0]):
    if i == 0:
        ttX = np.array(tt_X[2013].iloc[i, :]).reshape(1, 4)
        pred = rf.predict(ttX)
        ans_lst.extend(pred)
    else:
        _3 = ttX[0,2]
        ttX = np.array(tt_X[2013].iloc[i, :]).reshape(1, 4) # 다음 시간대 
        ttX[0][2] = pred 
        ttX[0][3] = _3
        pred = rf.predict(ttX)
        ans_lst.extend(pred)

weather2 = pd.read_csv('./data/기상2010_2019.csv', index_col = 0)
weather2.index = pd.DatetimeIndex(weather2.index)
temp = weather2[list(weather2.filter(regex = '기온'))]

len(temp['2010'])
len(temp['2011']) #
len(temp['2012']) # 
len(temp['2013'])
len(temp['2014'])
len(temp['2015'])
len(temp['2016'])

temp['2011'].index.hour.value_counts() # 18시 하나 없음 10월 12일
temp['2012'].index.hour.value_counts() # 2시 하나 없음 11월 29일




temp = addShift3(temp)
tr_X = temp[:'2018'].drop(['기온(°C)'], axis = 1); tr_y = temp[:'2018']['기온(°C)']
tt_X = temp['2019':].drop(['기온(°C)'], axis = 1); tt_y = temp['2019':]['기온(°C)']

ans_lst = []
rf = model.fit(tr_X, tr_y)

tt_X.iloc[0,:]
for i in range(tt_X[2013].shape[0]):
    if i == 0:
        ttX = np.array(tt_X.iloc[i, :]).reshape(1, 4)
        pred = rf.predict(ttX)
        ans_lst.extend(pred)
    else:
        _3 = ttX[0,2]
        ttX = np.array(tt_X[2013].iloc[i, :]).reshape(1, 4) # 다음 시간대 
        ttX[0][2] = pred 
        ttX[0][3] = _3
        pred = rf.predict(ttX)
        ans_lst.extend(pred)

"""
## Scaling
### Scaling

scaler = MinMaxScaler(feature_range=(1, 2))

_X = temp.drop(['기온(°C)'], axis = 1)
_X.columns # ['기온(°C)_day', '기온(°C)_year']
_y = pd.DataFrame(data['기온(°C)'])
_y.columns # ['기온(°C)']


scalerX = scaler.fit(_X)
scaled_X = scalerX.transform(_X) # 컬럼 참고 위해서

scalery = scaler.fit(_y)
scaled_y = scalery.transform(_y) # 컬럼 참고 위해서

### scaled data 합치기 
tf_data = pd.DataFrame(scaled_X, columns= _X.columns)
tf_data[_y.columns] = scaled_y
tf_data.index = idxes
"""
# =============================================================================
# TEST
# tr_X, tr_y, tt_X, tt_y = train_test_split(data = temp, length = 3, y_str = '기온(°C)')
# scr_dct = get_tr_tt_loss(tr_X, tr_y, tt_X, tt_y, model)
# =============================================================================
## 모델 설정
mdl_nmae = {}
#model = MLPRegressor(solver = 'lbfgs')
#model = LinearRegression()
#model = LGBMRegressor()
#model = XGBRegressor()
model = RandomForestRegressor()

## 최종 딕셔너리(3개월로 자른 데이터)

## 데이터 별로 nmae 얻기 
def tsCV_predict(data, y_str, model):
    nmae_mean_dct = {length+1:np.nan for length in range(6)}
    total_score_dct = {length+1:np.nan for length in range(6)}
    
    for i in range(6):
        tr_X, tr_y, tt_X, tt_y = train_test_split(data = data, length = i+1, y_str = y_str)
        
        scr_dct = get_tr_tt_mse_loss2019(tr_X, tr_y, tt_X, tt_y, model)
        total_score_dct[i+1] = scr_dct
    
        tr, tt = get_tr_tt_mean_loss(scr_dct)
        nmae_mean_dct[i+1] = pd.Series(data = [tr, tt], index = ["train_loss", "test_loss"])
        print("==="*15) 

        print(f"{i+1}년단위 스플릿 평균 mse")
        print(f" mean tr loss = {tr:.5f}, mean tt loss = {tt:.5f}")
        print("==="*15) 
    return nmae_mean_dct, total_score_dct
    

def plot_loss_per_split(nmae_mean_dct, y_str):
    tr_sr = list()
    tt_sr = list()
    for i in nmae_mean_dct.keys():
        tr_sr.append(nmae_mean_dct[i]['tr_loss'])
        tt_sr.append(nmae_mean_dct[i]['tt_loss'])
    tr_sr = pd.Series(tr_sr, index = list(nmae_mean_dct.keys()))
    tt_sr = pd.Series(tt_sr, index = list(nmae_mean_dct.keys()))

    sns.lineplot(data = tr_sr, x = tr_sr.index, y = tr_sr, marker = 'o', markersize = 12, lw =4.5,  label = 'train loss')
    sns.lineplot(data = tt_sr, x = tt_sr.index, y = tt_sr, marker = 'o', markersize = 12, lw =4.5, label = 'test loss')
    plt.legend(); plt.title(f'{model.__class__.__name__}모델 "{y_str}"예측 스플릿 별 loss', fontsize = 35, fontweight = 'bold')
    plt.plot()


def tsCV_predict(data, y_str, model):
    nmae_mean_dct = {length+1:np.nan for length in range(6)}
    total_score_dct = {length+1:np.nan for length in range(6)}
    
    for i in range(6):
        tr_X, tr_y, tt_X, tt_y = train_test_split(data = data, length = i+1, y_str = y_str)
        scr_dct = get_tr_tt_mse_loss2019(tr_X, tr_y, tt_X, tt_y, model)
        total_score_dct[i+1] = scr_dct
    
    
        tr, tt = get_tr_tt_mean_loss(scr_dct)
        nmae_mean_dct[i+1] = pd.Series(data = [tr, tt], index = ["train_loss", "test_loss"])
        print("==="*15) 

        print(f"{i+1}년단위 스플릿 평균 mse")
        print(f" mean tr loss = {tr:.5f}, mean tt loss = {tt:.5f}")
        print("==="*15) 
    return nmae_mean_dct, total_score_dct

# =============================================================================
# X변수별로 트래킹 
# =============================================================================

"""기온 예측 정확도 확인"""
def printTestLoss(total_score_dct):
    for i in total_score_dct.keys():
        print(f"---------------- {i} 스플릿 ----------------")
        for j in total_score_dct[i].keys():
            print(f"예측 시작 년도 {j}년")
            print(total_score_dct[i][j]['tt_loss'])
        
        
model = RandomForestRegressor()
model = RandomForestRegressor(n_estimators=400,min_samples_split=2,min_samples_leaf=4, max_features='sqrt',max_depth=10, bootstrap=True)

nmae_mean_dct, total_score_dct = tsCV_predict(temp, '기온(°C)', model)
printTestLoss(total_score_dct)

for key in total_score_dct.keys():
    plot_loss_per_split(total_score_dct[key], '기온(°C)')
    

"""30cm 지중온도 예측 정확도 확인"""
nmae_mean_dct, total_score_dct_30 = tsCV_predict(temp30, '30cm 지중온도(°C)', model)
for key in total_score_dct_30.keys():
    plot_loss_per_split(total_score_dct_30[key], '30cm 지중온도(°C)')
printTestLoss(total_score_dct_30)

        
nmae_mean_dct, total_score_dct_20 = tsCV_predict(temp20, '20cm 지중온도(°C)', model)
for key in total_score_dct.keys():
    plot_loss_per_split(total_score_dct[key], '20cm 지중온도(°C)')
printTestLoss(total_score_dct_20)
    
nmae_mean_dct, total_score_dct_10 = tsCV_predict(temp10, '10cm 지중온도(°C)', model)
for key in total_score_dct.keys():
    plot_loss_per_split(total_score_dct[key], '10cm 지중온도(°C)')
printTestLoss(total_score_dct_10)
    
nmae_mean_dct, total_score_dct_5 = tsCV_predict(temp5, '5cm 지중온도(°C)', model)
for key in total_score_dct.keys():
    plot_loss_per_split(total_score_dct[key], '5cm 지중온도(°C)')
printTestLoss(total_score_dct_5)
    
nmae_mean_dct, total_score_dct_gr = tsCV_predict(temp_gr, '지면온도(°C)', model)
for key in total_score_dct.keys():
    plot_loss_per_split(total_score_dct[key], '지면온도(°C)')
printTestLoss(total_score_dct_gr)
    
nmae_mean_dct, total_score_dct_ff = tsCV_predict(temp_f, '체감온도', model)
for key in total_score_dct.keys():
    plot_loss_per_split(total_score_dct[key], '체감온도')    
printTestLoss(total_score_dct_ff)
"""
2013년부터, target year = 2016
trian_loss = 12.59907, test_loss = 26.85344
"""
nmae_mean_dct, total_score_dct_hpa = tsCV_predict(temp_hpa, '증기압(hPa)', model)
for key in total_score_dct.keys():
    plot_loss_per_split(total_score_dct[key], '증기압(hPa)')   
printTestLoss(total_score_dct_hpa)

    
nmae_mean_dct, total_score_dct_drp = tsCV_predict(temp_drp, '이슬점온도(°C)', model)
for key in total_score_dct.keys():
    plot_loss_per_split(total_score_dct[key], '이슬점온도(°C)')
printTestLoss(total_score_dct_drp)



## 모델별 결과 저장  
mdl_nmae[model.__class__.__name__] = nmae_mean_dct
mdl_nmae

model = RandomForestRegressor()
nmae_mean_dct, total_score_dct = tsCV_predict(temp30, '30cm 지중온도(°C)', model)
plot_loss_per_split(nmae_mean_dct, '30cm 지중온도(°C)')

# =============================================================================
# SELECT MODEL AND PREDICT
# RF 
# TR SIZE = 2013-2018
# 예측 후 total_temp_pred_df.csv로 저장(2018 오리지널 2019 예측으로 대체한 temp 들어간 total df)
# =============================================================================
def predict2019(data, lng, y_str):
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = data, length =lng, y_str = y_str)
    pred = model.fit(tr_X[2013], tr_y[2013]).predict(tt_X[2016])
    pred = pd.Series(pred, index = tt_y[2016].index)

    mse = mean_squared_error(tt_y[2016], pred)
    pred.plot(label = 'pred'); tt_y[2016].plot(label = 'true');plt.legend(); plt.title(f"{y_str} 예측 with {model.__class__.__name__} mse = {mse:.5f}", fontsize = 30, fontweight = 'bold')
    return pred

model = RandomForestRegressor(n_estimators=400,min_samples_split=2,min_samples_leaf=4, max_features='sqrt',max_depth=10, bootstrap=True)
"""temp 예측"""


 tr_X, tr_y, tt_X, tt_y = train_test_split(data = temp, length =2, y_str = '기온(°C)')
 tr_X[2013], tr_y[2013], tt_X[2013], tt_y[2013] 
# =============================================================================
#  2019년 안들어간거 확인!!!
# tr_X, tr_y, tt_X, tt_y = train_test_split(data = temp, length =2, y_str = '기온(°C)')
# pred = model.fit(tr_X[2013], tr_y[2013]).predict(tt_X[2013])
# 
# =============================================================================

temp_pred = predict2019(temp,3, '기온(°C)')
temp_gr_pred = predict2019(temp_gr,3, '지면온도(°C)')
temp5_pred = predict2019(temp5,3, '5cm 지중온도(°C)')
temp10_pred = predict2019(temp10,3, '10cm 지중온도(°C)')
temp20_pred = predict2019(temp20,3, '20cm 지중온도(°C)')
temp30_pred = predict2019(temp30,3, '30cm 지중온도(°C)')
temp_f_pred = predict2019(temp_f,3, '체감온도')
temp_hpa_pred = predict2019(temp_hpa,3, '증기압(hPa)')
temp_drp_pred = predict2019(temp_drp,3, '이슬점온도(°C)')


cols= ['기온(°C)', '지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)', '체감온도', '증기압(hpa)',  '이슬점온도(°C)']
temp_pred_df = pd.DataFrame( columns = cols)
temp_pred_df[cols[0]] = temp_pred
temp_pred_df[cols[1]] = temp_gr_pred
temp_pred_df[cols[2]] = temp5_pred
temp_pred_df[cols[3]] = temp10_pred
temp_pred_df[cols[4]] = temp20_pred
temp_pred_df[cols[5]] = temp30_pred
temp_pred_df[cols[6]] = temp_f_pred
temp_pred_df[cols[6]] = temp_hpa_pred
temp_pred_df[cols[6]] = temp_drp_pred

## 저장 
temp_pred_df.to_csv('./data/temp_13_15_pred_df.csv')

temp_pred_df.columns
_ = pd.read_csv('./data/total_df.csv', index_col = 0)
_.index = pd.DatetimeIndex(_.index)

for comp in _['구분'].unique():
    _.query(f"구분 == '{comp}'").loc['2019':, cols] = temp_pred_df 
    
#_.to_csv('./data/total_temp_13_15_pred_df.csv')
