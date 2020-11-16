#!/usr/bin/env python
# coding: utf-8

# In[1]:



get_ipython().run_line_magic('matplotlib', 'notebook')
import pandas as pd
import numpy as np
import math
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

data = pd.read_csv(r"C:\Users\USER\.jupyter\TSLA2.csv")


# In[3]:


data.set_index('Date', inplace = True)
print(data.head (2))


# In[4]:


data['HL_PCT']= (data['High']-data['Low'])/data['Low']
data['PCT_change']= (data['Adj Close']-data['Open'])/data['Open']
print(data.head(2))


# In[5]:


data = data[['HL_PCT','PCT_change','Adj Close','Volume']]


# In[6]:


print(data.head(2))


# In[7]:


forecast_col = 'Adj Close'
forecast_out = math.ceil(0.01* len(data))   
print(forecast_out)
data.fillna(-99999, inplace = True)
data['label']= data[forecast_col].shift(-forecast_out)


# In[8]:


X = np.array(data.drop(['label'], 1))
X = X[:-forecast_out]
X_predict = X[-forecast_out:]
y = np.array(data['label'])
y = y[:-forecast_out]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
clf = LinearRegression(n_jobs = -1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print (accuracy)
forecast_value = clf.predict(X_predict)
print(forecast_value, accuracy)


# In[9]:


data['Forecast'] = np.nan
last_date_ = data.iloc[-1].name
last_date = datetime.datetime.strptime(last_date_,"%Y-%m-%d" )
print (last_date)
last_unix = last_date.timestamp()
print (last_unix)


# In[10]:


one_day = 24*60*60    
next_unix = last_unix + one_day

for i in forecast_value:
    Date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data.loc[Date] = [np.nan for all in range(len(data.columns)-1)] + [i]
    
print (data.head())
   
data['Adj Close'].plot()
data['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Dates')
plt.ylabel ('Price')
plt.show()


# In[ ]:




