#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


motor=pd.read_csv('pmsm_temperature_data.csv')


# In[7]:


motor.head()


# In[8]:


sns.heatmap(motor.isnull())


# In[9]:


motor.drop(['profile_id'],axis=1,inplace=True)


# In[10]:


motor.head()


# In[11]:


from sklearn.model_selection import train_test_split


# In[12]:


X_train,X_test,y_train,y_test=train_test_split(motor.drop('stator_yoke',axis=1),
                                               motor['stator_yoke'],
                                               test_size=0.30,random_state=101)
X_train,X_test,z_train,z_test=train_test_split(motor.drop('stator_tooth',axis=1),
                                               motor['stator_tooth'],
                                               test_size=0.30,random_state=101)
X_train,X_test,a_train,a_test=train_test_split(motor.drop('stator_winding',axis=1),
                                               motor['stator_winding'],
                                               test_size=0.30,random_state=101)


# In[34]:


from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train,y_train)
lm.fit(X_train,z_train)
lm.fit(X_train,a_train)


# In[36]:


print(lm.intercept_)


# In[37]:


lm.coef_


# In[38]:


predictions=lm.predict(X_test)


# In[39]:


plt.scatter(y_test,predictions)


# In[40]:


plt.scatter(z_test,predictions)


# In[41]:


plt.scatter(a_test,predictions)


# In[42]:


from sklearn import metrics


# In[43]:


metrics.mean_absolute_error(y_test,predictions)


# In[44]:


metrics.mean_absolute_error(z_test,predictions)


# In[45]:


metrics.mean_absolute_error(a_test,predictions)

