#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_csv(r"C:\Users\USER\.jupyter\TSLA2.csv", index_col ="Date", parse_dates = True)
import os


# In[2]:


data.shape


# In[3]:


data.head()


# In[4]:


plt.figure(figsize=(20, 15))

plt.subplot(2,1,1)
plt.plot(data['Adj Close'], label='Adj Close', color="purple")
plt.legend(loc="upper right")
plt.title('Adj Close Prices of Tesla')

plt.subplot(2,1,2)
plt.plot(data['Volume'], label='Volume', color="Orange")
plt.legend(loc="upper right")
plt.title('Volume Of Shares Traded')


# In[5]:


get_ipython().system('pip install pmdarima ')


# In[6]:


from pmdarima import auto_arima


# In[7]:


import warnings 
warnings.filterwarnings("ignore") 
stepwise_fit = auto_arima(data['Adj Close'], start_p = 1, start_q = 1, 
                          max_p = 3, max_q = 3, m = 12, 
                          start_P = 0, seasonal = True, 
                          d = None, D = 1, trace = True, 
                           error_action ='ignore', 
                          suppress_warnings = True,
                          stepwise = True)
stepwise_fit.summary() 


# In[10]:


train = data.iloc[:len(data)-12] 
test = data.iloc[len(data)-12:]
from statsmodels.tsa.statespace.sarimax import SARIMAX 
  
model = SARIMAX(train['Adj Close'],  
                order = (1, 0, 2),  
                seasonal_order =(0, 1, 1, 12)) 
  
result = model.fit() 
result.summary() 


# In[11]:


start = len(train) 
end = len(train) + len(test) - 1
predictions = result.predict(start, end, 
                             typ = 'levels').rename("Predictions") 
predictions_df = pd.DataFrame(predictions)
predictions_df.index = ['2020-01-16', '2020-01-17', '2020-01-21', '2020-01-22',
               '2020-01-23', '2020-01-24', '2020-01-27', '2020-01-28',
               '2020-01-29', '2020-01-30', '2020-01-31', '2020-02-03']
predictions_df.index = pd.to_datetime(predictions_df.index)
predictions_df.plot(legend = True) 
test['Adj Close'].plot(legend = True)


# In[12]:


from sklearn.metrics import mean_squared_error 
from statsmodels.tools.eval_measures import rmse 
print("RMSE on Test Data: ", rmse(test["Adj Close"], predictions))
print("MSE on Test Data: ", mean_squared_error(test["Adj Close"], predictions))
model = model = SARIMAX(data['Adj Close'],  
                        order = (1, 0, 2),  
                        seasonal_order =(0, 1, 1, 12)) 
    
result = model.fit() 
forecast = result.predict(start = len(data),  
                          end = (len(data)-1) + 1,
                          typ = 'levels').rename('Forecast') 
print("The predicted share price on the 1rd May 2019 is: {}".format(forecast.iloc[0]))


# In[ ]:




