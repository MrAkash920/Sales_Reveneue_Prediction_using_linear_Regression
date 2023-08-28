#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')


# In[13]:


df = pd.read_csv('Advertising Budget and Sales.csv', encoding='latin1')
df.head(10)


# In[14]:


df.describe(include='all')


# In[15]:


df.info()


# In[16]:


df.shape


# In[17]:


df.ndim


# In[18]:


df.size


# In[19]:


pd.DataFrame(df.isnull().sum(),columns =["Count of Null Values"]).T


# In[21]:


a = df["Newspaper Ad Budget ($)"]
b = df["TV Ad Budget ($)"]


# In[22]:


sns.scatterplot(x=a,y=b,color='purple')


# In[23]:


sns.distplot(df['Radio Ad Budget ($)'])


# In[25]:


sns.pairplot(df)


# In[27]:


sns.pairplot(df,x_vars=['Newspaper Ad Budget ($)','Radio Ad Budget ($)','TV Ad Budget ($)'],y_vars=['Sales ($)'],height=3,aspect=1)
plt.show()


# In[28]:


#linear Regression
X = df[['TV Ad Budget ($)']]
Y = df['Sales ($)']


# In[57]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.2,random_state=48)


# In[58]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,Y_train)


# In[59]:


lr.intercept_


# In[60]:


lr.coef_


# In[61]:


print("The LR model is: Y = ",lr.intercept_, "+", lr.coef_, "TV Ad Budget ($)")


# In[62]:


lr.score(X_train,Y_train)


# In[63]:


Y_pred =lr.predict(X_test)


# In[64]:


Y_pred


# In[65]:


diff = pd.DataFrame({'Actual':Y_test,'Predicted':Y_pred})


# In[66]:


diff.head(10)


# In[67]:


from sklearn import metrics
from sklearn.metrics import r2_score


# In[68]:


R2 = r2_score(Y_test,Y_pred)
mae = metrics.mean_absolute_error(Y_test,Y_pred)
mse = metrics.mean_squared_error(Y_test,Y_pred)
rmse = np.sqrt(metrics.mean_squared_error(Y_test,Y_pred))


# In[69]:


print('Accuracy = ',R2.round(2)*100,'%')
print('mae =',mae.round(2))
print('mse =',mse.round(2))
print('rmse =',rmse.round(2))


# In[70]:


sns.regplot(x=Y_test, y=Y_pred,color='green')


# In[ ]:




