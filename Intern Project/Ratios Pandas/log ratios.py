# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 10:15:06 2018

@author: ashwin.monpur
"""

import pandas as pd
import math
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import datetime
import numpy as np

MF_Data=pd.read_csv('Book1.csv')
nav=MF_Data['Net Asset Value']

nav.index=pd.to_datetime(MF_Data['Date'])

# Get index Data from csv
index_data = pd.read_csv('nif50ty.csv')

#index_open=index_data['Open']
index_close=pd.to_numeric(index_data['Close'])
#index_avg=(index_open+index_close)/2
index_close.index=pd.to_datetime(index_data['Date'])

date_data=np.max(pd.to_datetime(MF_Data['Date']))
fdat_a=pd.to_datetime(date_data - relativedelta(months=24))

start=pd.to_datetime(datetime.datetime.strftime(fdat_a,'%Y-%m-%d'))
end=pd.to_datetime(datetime.datetime.strftime(date_data,'%Y-%m-%d'))
step=datetime.timedelta(days=1)

nav_last2_years=nav[start:end]
index_close_last2_years=index_close[start:end]


#plt.subplot(1,2,1)
#plt.plot(nav)
#plt.plot(index_close)

list_av_mf=[]
list_match_indx=[]
#list_mf_d_date=[]
#list_indx_d_date=[]
#
#
for i in pd.to_datetime(nav_last2_years.index):
    for j in pd.to_datetime(index_close_last2_years.index):
#        print(i)
#        print(j)
        list_mf=[]
        list_indx=[]
        list_mf_d=[]
        list_indx_d=[]
        if i==j:
            list_mf.append(j)
            list_mf.append(nav_last2_years[j])
            list_indx.append(i)
            list_indx.append(index_close_last2_years[i])
            
            list_av_mf.append(list_mf)
            list_match_indx.append(list_indx)
#            list_mf_d_date.append(list_mf_d)
#            list_indx_d_date.append(list_indx_d)
#            print('index: {0} {1}'.format(i,index_ret[i]))
#            print('MF   : {0} {1}'.format(j,ret_d[j]))
        else:
            continue
#
#
nav_spd_data=pd.DataFrame(list_av_mf)
indx_spd_data=pd.DataFrame(list_match_indx)

nav_spd_data.columns=['Date','NAV']
indx_spd_data.columns=['Date','Close']
nav_spd_data.index=nav_spd_data['Date']
indx_spd_data.index=indx_spd_data['Date']

data_nav=nav_spd_data['NAV']
data_indx=indx_spd_data['Close']
#
#
risk_free_rate=6.9/365

ratios_mf={}

# Using log for returns and calculating ratios

nav_returns = np.log(data_nav/data_nav.shift(1))

# volatility

nav_volt=pd.rolling_std(nav_returns,window=252)*np.sqrt(252)

# Sharpe ratio

def sharpe_ratio(returns,rf,days=252):
    volatility = returns.std() * np.sqrt(days) 
    sharpe_ratio = (returns.mean() - rf) / volatility
    return sharpe_ratio

sr=sharpe_ratio(nav_returns, risk_free_rate)
