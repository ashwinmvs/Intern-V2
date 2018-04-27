# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 14:53:26 2018

@author: ashwin.monpur
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA
import itertools
import statsmodels.api as sm

data=pd.read_csv('MFC_3.csv')

ts_nav=data['nav']

date_nav = pd.to_datetime(data['date']) #dtype: datetime64[ns] Convert argument to datetime.
ts_nav.index=date_nav
#plt.plot(ts_nav)

# Minimizing the variance

ts_log=np.log(ts_nav)
#plt.plot(ts_log)  #varaince of the data is minimized

# making the data stationary using 1st order differencing

ts_logdff= ts_log-ts_log.shift()
ts_logdff.dropna(inplace=True) #removing NaN values 
#plt.plot(ts_logdff)

# According to the plot, seasonality exists so 2nd order differencing is needed

ts_log2dff= ts_logdff-ts_logdff.shift()
ts_log2dff.dropna(inplace=True)
#plt.plot(ts_log2dff)

#Checking seasonal & differencing for stationarity 
# Dickey-Fuller (ADF) test

def test_stationarity(timeseries):

#Perform Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC') 
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
test_stationarity(ts_logdff)

#p = d = q = range(0, 2) # Define the p, d and q parameters to take any value between 0 and 2
#pdq = list(itertools.product(p, d, q)) # Generate all different combinations of p, q and q triplets
#pdq_x_QDQs = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))] # Generate all different combinations of seasonal p, q and q triplets
#
#AIC_dic={}
#for param in pdq:
#    for seasonal_param in pdq_x_QDQs:
#        try:
#            mod = sm.tsa.statespace.SARIMAX(ts_nav,
#                                            order=param,
#                                            seasonal_order=seasonal_param,
#                                            enforce_stationarity=False,
#                                            enforce_invertibility=False)
#            results = mod.fit()
#            key='SARIMA{}x{}'.format(param,seasonal_param)
#            AIC_dic.setdefault(key,results.aic)
#        except:
#            continue
#print()
#print(min(AIC_dic.values()))
#print(min(AIC_dic.items(), key=lambda x: x[1])[0])

mod = sm.tsa.statespace.SARIMAX(ts_nav, 
                                order=(1,1,1), 
                                seasonal_order=(0,1,1,3),   
                                enforce_stationarity=False,
                                enforce_invertibility=False)
results = mod.fit()
prediction=results.get_prediction(start='2013-03-01')
pred_ci = prediction.conf_int()


#ax.set_ylabel("Net Asset Value")
#plt.legend()
#sns.despine()
#act_pred=np.exp(prediction)

print(results.summary())

#results = np.exp(results.fittedvalues)
#plt.subplot(1,2,1)
#plt.plot(ts_log)
#plt.plot(results.fittedvalues)
#
#
#plt.subplot(1,2,2)
#plt.plot(predictions_ARIMA.shift())
#plt.plot(ts_nav)


data_test=pd.read_csv('test_MFC.csv')

test = data_test['nav']
test_date = pd.to_datetime(data_test['date'])
test.index=test_date

forecast = results.forecast(steps = 100)
predicted_list = np.exp(forecast[0])
#f_prediction_data=np.exp(predicted_list)

dates=pd.date_range('2015-08-12', periods=100)

plt.plot(results.fittedvalues)
ax = ts_nav.plot(label='Observed')
prediction.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
#plt.subplot(2,1,1)
#plt.plot(ts_nav)
plt.plot(dates,predicted_list)
plt.plot(ts_nav)
plt.plot(test)

