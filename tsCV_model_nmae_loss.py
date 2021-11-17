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
os.chdir('../Gas_Prediction') # on window
# os.chdir('./DS_Projects/Gas_Prediction') # on mac

data = pd.read_csv('./data/total_train.csv', index_col = 0)
data.index = pd.DatetimeIndex(data.index)
# A = data.query("구분 == 'A'").drop(['구분'], axis = 1)


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
        print(f"{yr}년부터  {score}")
        
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
        print(f"{yr}년부터  {sc}")
        
    return score_lst

# =============================================================================
# TEST
# model = RandomForestRegressor()
# res = get_mean_nmae(tr_X, tr_y, te_X, te_y, model)
# res
# =============================================================================

model = LinearRegression()
model = MLPRegressor(solver = 'lbfgs')
nmae_mean_dct = {length+1:np.nan for length in range(5)}

for i in range(5):
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = A, length = i+1)
    score = get_mean_nmae(tr_X, tr_y, tt_X, tt_y, model)
    print(f"A 공급사의 {i+1}년단위 스플릿 nmae 평균 = {score}")
    nmae_mean_dct[i+1] = score


# =============================================================================
# 전체 공급사에 적용 
#  - 3개월치로 자른 데이터
# =============================================================================
## 데이터 불러오기
data = pd.read_csv('./data/total_train.csv', index_col = 0)
data.index = pd.DatetimeIndex(data.index)

## 모델 설정
mdl_nmae = {}
model = MLPRegressor(solver = 'lbfgs')
model = LinearRegression()
model = LGBMRegressor()
model = XGBRegressor()
model = RandomForestRegressor()



## 최종 딕셔너리(3개월로 자른 데이터)
comp_nmae_dct = {comp:pd.DataFrame() for comp in data['구분'].unique()}
for comp in list(data['구분'].unique()):
    ## 데이터셋 나누기
    comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
    comp_data = comp_data[(comp_data.index.month == 1)|(comp_data.index.month == 2)|(comp_data.index.month == 3)]
    nmae_mean_dct = {length+1:np.nan for length in range(5)}
    ## 데이터 별로 nmae 얻기 
    print(f"=======  {comp} 공급사  ========")
    for i in range(5):
        tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = i+1)
        score = get_mean_nmae(tr_X, tr_y, tt_X, tt_y, model)
        print(f"{comp} 공급사의 {i+1}년단위 스플릿 nmae 평균 = {score}")
        nmae_mean_dct[i+1] = score
    print("==="*10)    
    comp_nmae_dct[comp] = nmae_mean_dct

## 모델별 결과 저장  
mdl_nmae_3mth[model.__class__.__name__] = comp_nmae_dct
# =============================================================================
# ## 최종 모델 정보 저장
# 
# with open('./data/mdl_nmae_3mth.json', 'w') as fp:
#     json.dump(mdl_nmae_3mth, fp)
# 
# ## 최종 모델 정보 저장한거 불러오기
# with open('./data/mdl_nmae_3mth.json') as json_file:
#     mdl_nmae_3mth = json.load(json_file)
# =============================================================================

tt = []
for comp in data['구분'].unique():
    print(min(mdl_nmae['MLPRegressor'][comp].values()))    
    tt.append(min(mdl_nmae['MLPRegressor'][comp].values()))
pd.Series(tt).mean()    
# =============================================================================
# 결과조회
# =============================================================================

## 결과조회 (전체)
with open('./data/mdl_nmae.json') as json_file:
    mdl_nmae = json.load(json_file)
print("used whole dataset")
for comp in data['구분'].unique():
    print(f"======================== {comp} 공급사 ========================")
    test = pd.Series()
    for mdl in mdl_nmae.keys():
        
        val = min(mdl_nmae[mdl][comp].values())
        #print(f"{item}모델에서 최소 nmae = {val:.4f}")
        test[mdl]= val
        
    print(f"optimal 모델: {test[test == test.min()].index[0]} with nmae = {test.min():.4f}")

test.mean()
## 결과 조회 (3개월)
print('used three months (01, 02, 03)')
with open('./data/mdl_nmae_3mth.json') as json_file:
     mdl_nmae_3mth = json.load(json_file)

for comp in data['구분'].unique():
    print(f"======================== {comp} 공급사 ========================")
    test = pd.Series()
    for mdl in mdl_nmae_3mth.keys():
        
        val = min(mdl_nmae_3mth[mdl][comp].values())
        #print(f"{item}모델에서 최소 nmae = {val:.4f}")
        test[mdl]= val
        
    print(f"optimal 모델: {test[test == test.min()].index[0]} with nmae = {test.min():.4f}")

# =============================================================================
# 결과 시각화 
# =============================================================================

import matplotlib as mpl
print (mpl.matplotlib_fname())
## 시각화
import matplotlib.pyplot as plt
import seaborn as sns
# pip install matplotlib --upgrade
fig, ax = plt.subplots(nrows = 2, ncols =4, sharex=True, sharey = True)
ax= ax.reshape(8)

for idx, comp in enumerate(data['구분'].unique()):
    sns.lineplot(x = comp_nmae_dct[comp].keys(), y = comp_nmae_dct[comp].values(), ax = ax[idx], marker = 'o')
    ax[idx].set_title(f"{comp} 공급사", loc = 'left')
plt.suptitle("MLP with solver lbfgs NMAEs per comp")
plt.supxlabel("split 단위"); plt.supylable("평균 nmae")
# =============================================================================
# 제일 잘나왔던 모델 test loss (nmae) 계산 
# =============================================================================

model = RandomForestRegressor()
tt_lst = []
for comp in list(data['구분'].unique()):
    comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
    
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_mean_nmae(tr_X, tr_y, tt_X, tt_y, model)
    tt_lst.append(score)
    print(f"{comp} 공급사 = {score}")
print(np.mean(tt_lst))    



# =============================================================================
# MLP 자세히 조사 
# length = 5 로 했을 때 (2013~2017, 2014~2018)
# =============================================================================

## 각 공급사 별 모델들의 test loss 도출
test =[]
model = MLPRegressor()
for comp in list(data['구분'].unique()):
    ## 데이터셋 나누기
    comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
    #comp_data = comp_data[(comp_data.index.month == 1)|(comp_data.index.month == 2)|(comp_data.index.month == 3)]
    nmae_mean_dct = {length+1:np.nan for length in range(5)}
    ## 데이터 별로 nmae 얻기 
    print(f"=======  {comp} 공급사  ========")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_mean_nmae(tr_X, tr_y, tt_X, tt_y, model)
    print(f"{comp} 공급사의 5년단위 스플릿 nmae 평균 = {score}")
    test.append(score)
print(np.mean(score))
    
## 각 공급사 별 모델들의 tr, tt loss 도출
## under, overfitting 정도 알기 위해
scr_lst = {comp:np.nan for comp in data['구분'].unique()}
model = LGBMRegressor()
for comp in list(data['구분'].unique()):
    ## 데이터셋 나누기
    comp_data = data.query(f"구분 == '{comp}'").drop(['구분'], axis = 1)
    #comp_data = comp_data[(comp_data.index.month == 1)|(comp_data.index.month == 2)|(comp_data.index.month == 3)]
    nmae_mean_dct = {length+1:np.nan for length in range(5)}
    ## 데이터 별로 nmae 얻기 
    print(f"=======  {comp} 공급사  ========")
    tr_X, tr_y, tt_X, tt_y = train_test_split(data = comp_data, length = 5)
    score = get_tr_tt_loss(tr_X, tr_y, tt_X, tt_y, model)
    print(score)
    scr_lst[comp] = score

for comp in scr_lst.keys():
    print(f'{comp} 공급사')
    print(scr_lst[comp])
    
"""
<결과 비교>
<================ MLP (maxiter = 3000) ================> - tr loss > tt loss
A 공급사
[tr_loss    0.110355  tt_loss    0.076471
B 공급사
[tr_loss    0.134156  tt_loss    0.091515
C 공급사
[tr_loss    0.340538  tt_loss    0.098154
D 공급사
[tr_loss    0.120640  tt_loss    0.082725
E 공급사
[tr_loss    0.119712  tt_loss    0.103506
G 공급사
[tr_loss    0.086687  tt_loss    0.106139
H 공급사
[tr_loss    0.125292  tt_loss    0.08329


<================ MLP (maxiter = 2000) ================> - tr loss > tt loss
A 공급사
[tr_loss    0.112430  tt_loss    0.076673
B 공급사
[tr_loss    0.132437  tt_loss    0.091837
C 공급사
[tr_loss    0.342535  tt_loss    0.096725
D 공급사
[tr_loss    0.129711  tt_loss    0.082480
E 공급사
[tr_loss    0.117635  tt_loss    0.103062
G 공급사
[tr_loss    0.087699  tt_loss    0.105342
H 공급사
[tr_loss    0.127505  tt_loss    0.084730

MLP는 학습을 잘 못함 

<================ RandomForest ================> - tr loss < tt loss
A 공급사
[tr_loss    0.023485  tt_loss    0.083188
B 공급사
[tr_loss    0.027472  tt_loss    0.095853
C 공급사
[tr_loss    0.091936  tt_loss    0.111950
D 공급사
[tr_loss    0.026100  tt_loss    0.083801
E 공급사
[tr_loss    0.024721  tt_loss    0.108713
G 공급사
[tr_loss    0.018864  tt_loss    0.105817
H 공급사
[tr_loss    0.026577  tt_loss    0.085229

<================ XGBRegressor ================> - tr loss < tt loss
A 공급사
[tr_loss    0.054818  tt_loss    0.080327
B 공급사
[tr_loss    0.063642  tt_loss    0.096266
C 공급사
[tr_loss    0.219823  tt_loss    0.100450
D 공급사
[tr_loss    0.059567  tt_loss    0.083737
E 공급사
[tr_loss    0.057705  tt_loss    0.109748
G 공급사
[tr_loss    0.044179  tt_loss    0.102851
H 공급사
[tr_loss    0.062004  tt_loss    0.087398

<================ LGBRegressor ================> - tr loss < tt loss
A 공급사
[tr_loss    0.065879  tt_loss    0.076211
B 공급사
[tr_loss    0.076691  tt_loss    0.092345
C 공급사
[tr_loss    0.284360  tt_loss    0.097447
D 공급사
[tr_loss    0.072848  tt_loss    0.080383
E 공급사
[tr_loss    0.070482  tt_loss    0.105779
 G 공급사
[tr_loss    0.053358  tt_loss    0.104085
H 공급사
[tr_loss    0.074859  tt_loss    0.081862

"""