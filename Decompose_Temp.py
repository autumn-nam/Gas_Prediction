# =============================================================================
# 라이브러리로드 및 세팅 
# =============================================================================

import os 
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.forecasting.stl import STLForecast


os.chdir('../Gas_Prediction')
# =============================================================================
# 데이터 로드
# =============================================================================
## train set 준비 
total_df = pd.read_csv('./data/total_df.csv' , index_col = 0)
total_df.index = pd.DatetimeIndex(total_df.index)
_= total_df[total_df['구분'] == 'A'].drop(['구분'], axis = 1) # 온도는 공급사별로 같음 그래서 하나만 
_.columns
# 온도 관련 컬럼 추출
temp_cols = list(_.filter(regex = '온도')) + list(_.filter(regex = '기압'))+ list(_.filter(regex = '기온')) 
temp_cols # ['이슬점온도(°C)', '지면온도(°C)', '5cm 지중온도(°C)', '10cm 지중온도(°C)', '20cm 지중온도(°C)', '30cm 지중온도(°C)', '체감온도', '증기압(hPa)', '기온(°C)']
temps_df = _[temp_cols]
## 온도 관련 total temps df 저장 
# temps_df.to_csv('./data/total_temps_df.csv')

temp = temps_df['기온(°C)']
temp_gr = temps_df['지면온도(°C)']
temp5 = temps_df['5cm 지중온도(°C)']
temp10 = temps_df['10cm 지중온도(°C)']
temp20 = temps_df['20cm 지중온도(°C)']
temp30 = temps_df['30cm 지중온도(°C)']
temp_f = temps_df['체감온도']

# =============================================================================
# Descriptive Statistics
# =============================================================================
temp.describe()
temp.plot()
"""
       30cm 지중온도(°C)
count  366611.000000
mean       14.361431
std         9.400282
min        -1.200000
25%         5.200000
50%        15.400000
75%        23.300000
max        30.700000
"""
temp['2013'].plot()
# temp 하루 주기 확인 

daily_idx = list(range(0, len(temp['2013'].index), 24))
temp['2013-01-01':'2013-01-08'].plot(); plt.title('temp 하루주기 - 2013-01 첫째주', fontsize = 30, fontweight = 'bold')
plt.vlines(x = temp.index[daily_idx], ymin = temp.min(), ymax = temp.max(), color = 'coral')

temp['2013-08-01':'2013-08-08'].plot(); plt.title('temp 하루주기 - 2013-08 첫째주', fontsize = 30, fontweight = 'bold')
plt.vlines(x = temp.index[daily_idx], ymin = temp.min(), ymax = temp.max(), color = 'coral')

# temp 주단위 주기
weekly_idx = list(range(0, len(temp['2013'].index), 7*24))
temp['2013-01':'2013-03'].plot(); plt.title('2013-01~02월', fontsize = 30, fontweight = 'bold'); plt.vlines(x = temp.index[weekly_idx_2013], ymin = temp.min(), ymax = temp.max(), color = 'coral')
temp['2013-08': '2013-09'].plot(); plt.title('2013-08~09월', fontsize = 30, fontweight = 'bold'); plt.vlines(x = temp.index[weekly_idx_2013], ymin = temp.min(), ymax = temp.max(), color = 'coral')

# temp 반년 주기 확인
temp['2013-01':'2013-06'].plot(); plt.title('temp 반년주기', fontsize = 30, fontweight = 'bold')
temp['2013-06':'2013-12'].plot(); plt.title('temp 반년주기', fontsize = 30, fontweight = 'bold')
temp['2014-01':'2014-06'].plot(); plt.title('temp 반년주기', fontsize = 30, fontweight = 'bold')
temp['2014-06':'2014-12'].plot(); plt.title('temp 반년주기', fontsize = 30, fontweight = 'bold')
temp['2015-01':'2015-06'].plot(); plt.title('temp 반년주기', fontsize = 30, fontweight = 'bold')
temp['2015-06':'2015-12'].plot(); plt.title('temp 반년주기', fontsize = 30, fontweight = 'bold')

temp['2013-06-01':'2013-12-01'].plot(); plt.title('temp 반년주기', fontsize = 30, fontweight = 'bold')
temp['2014-01-01':'2014-06-01'].plot(); plt.title('temp 반년주기', fontsize = 30, fontweight = 'bold')

"""
Conclusion
- 일주단위 패턴 없음
- daily, yearly (하루 해지고 뜨는것, 사계절로 인한 영향) 뚜렷
"""
# =============================================================================
# Seasonal Decomposition (additive)  "temp  = 기온"
# =============================================================================
from statsmodels.tsa.seasonal import seasonal_decompose

total_temps_df = pd.read_csv('./data/total_temps_df.csv', index_col = 0)
total_temps_df.index = pd.DatetimeIndex(total_temps_df.index)
temps_dec_df = total_temps_df
temps_dec_df.columns

for col in temps_df.columns:
    print(col)
    res = seasonal_decompose(temps_df[col], period = 24*365); res.plot();
    res2 = seasonal_decompose(res.seasonal, period = 24); res.plot()
    temps_dec_df[str(col)+'_day'] = res2.seasonal
    temps_dec_df[str(col)+'_year'] = res2.trend
    
## year NA 대체
for col in list(temps_dec_df.filter(regex = 'year')):
    temps_dec_df[col]['2013-01-01'][:12] = temps_dec_df[col]['2014-01-01'][:12]
    temps_dec_df[col]['2019-03-31'][12:] = temps_dec_df[col]['2018-03-31'][12:] 

## NA 확인
temps_dec_df.isna().sum()
temps_dec_df[list(temps_dec_df.filter(regex = '이슬'))]
## 최종저장
#temps_dec_df.to_csv('./data/total_temps_dec_df.csv')
temps_dec_df.columns



# =============================================================================
# Residual 정규성 테스
# =============================================================================
# shapiro test
# - null: x ~ N(0, 1)
from scipy import stats
res = seasonal_decompose(temps_df['기온(°C)'], period = 24*365); res.plot();
res2 = seasonal_decompose(res.seasonal, period = 24); res2.plot()

test_res = stats.shapiro(res.resid)
print(test_res) # fail to reject the null
plt.vlines(res.resid.mean(), ymin = 0, ymax=1500, lw =5, colors = 'coral' , label = 'mean')
sns.histplot(res.resid); plt.title("365일 period dec 후 resid", fontweight = 'bold', fontsize = 30)


test_res = stats.shapiro(res2.resid)
print(test_res) # fail to reject the null
res2.resid.mean()
plt.vlines(res2.resid.mean(), ymin = 0, ymax=1500, lw =5, colors = 'coral' , label = 'mean')
sns.histplot(res.resid);  plt.title("365일 period dec + 24 period dec 후 resid", fontweight = 'bold', fontsize = 30)




from statsmodels.tsa.stattools import adfuller
result = adfuller(res.resid.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# reject the null (non-stationary) = stationary
 
# =============================================================================


# =============================================================================


