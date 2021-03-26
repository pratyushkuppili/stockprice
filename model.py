#!/usr/bin/env python
# coding: utf-8

# In[159]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime


# In[160]:


train_raw = pd.read_csv('Train.csv')
train_raw.head()


# In[161]:


train_raw[train_raw.duplicated()]


# In[162]:


train_raw.drop_duplicates(keep = 'first', inplace = True)


# In[163]:


train_raw.head()


# In[164]:


train_raw.drop(['InvoiceDate'],1,inplace = True)


# In[165]:


train_raw.head()


# In[166]:


train_raw.groupby(['InvoiceNo'])['UnitPrice'].sum()


# In[167]:


#plt.figure(figsize=(12,12))
#sns.pairplot(train_raw)


# In[168]:


sns.heatmap(train_raw.corr())


# In[169]:


train_raw.corr()


# In[170]:


plt.show(sns.boxplot(train_raw['Quantity']))
plt.show(sns.boxplot(train_raw['UnitPrice']))


# In[171]:


quantity_cutoff = train_raw['Quantity'].mean() + 3*train_raw['Quantity'].std()
quantity_cutoff


# In[172]:


train_raw = train_raw[train_raw['Quantity'] < quantity_cutoff]
train_raw


# In[173]:


unitprice_cutoff = train_raw['UnitPrice'].quantile(0.99)
unitprice_cutoff


# In[174]:


train_raw = train_raw[train_raw['UnitPrice'] <= unitprice_cutoff]
train_raw


# In[175]:


train_raw['UnitPrice'].quantile(0.99)


# In[176]:


plt.show(sns.boxplot(train_raw['UnitPrice']))


# In[ ]:





# In[177]:


train_raw.drop(list(train_raw[train_raw['Quantity']<0].index),inplace = True)


# In[178]:


train_raw['Quantity'].describe()


# In[179]:


plt.show(sns.boxplot(train_raw['Quantity']))


# In[180]:


plt.show(sns.boxplot(train_raw['UnitPrice']))


# In[181]:


train_raw.shape


# In[182]:


train_raw.head()


# In[183]:


target = train_raw['UnitPrice']
train_raw.drop(['UnitPrice'],1,inplace = True)


# In[184]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_raw,target,test_size = 0.3, random_state = 42)


# In[185]:


X_train.head()


# In[186]:


from sklearn.ensemble import RandomForestRegressor

import warnings
warnings.filterwarnings("ignore")


# In[187]:


#n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 3)]
#max_features = ['auto', 'sqrt']
#max_depth = [int(x) for x  in np.linspace(start = 5, stop = 30, num= 3)]
#min_samples_split = [2,5,10]
#min_samples_leaf = [5,10]


# In[156]:


rf_rand = RandomForestRegressor()
#from sklearn.model_selection import RandomizedSearchCV


# In[157]:


#random_grid = {"n_estimators":n_estimators,
#              "max_features":max_features,
#              "max_depth":max_depth,
#               "min_samples_split":min_samples_split,
#              "min_samples_leaf":min_samples_leaf}
#print(random_grid)


# In[192]:


#rf_model = RandomizedSearchCV(estimator = rf_rand, param_distributions = random_grid, 
#                   scoring ='neg_mean_squared_error', n_iter = 5, cv = 5, verbose = 2,
#                  random_state = 42, n_jobs = 1)


# In[190]:


rf_model = RandomForestRegressor(n_estimators= 500, max_features= 'auto', max_depth = 5, min_samples_split= 5, min_samples_leaf= 5 )


# In[191]:


rf_model.fit(X_train,y_train)


# In[193]:


predictions = rf_model.predict(X_test)
predictions


# In[195]:


sns.distplot(y_test-predictions)


# In[196]:


plt.scatter(y_test, predictions)


# In[198]:


import pickle
file = open('best_regression_model.pkl', 'wb')
pickle.dump(rf_model, file)


# In[ ]:




