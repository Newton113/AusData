#!/usr/bin/env python
# coding: utf-8

# # Importing libraries

# In[39]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.preprocessing import LabelEncoder


# # Importing data

# In[40]:


data = pd.read_csv('C:/Users/Prince Newton/Downloads/weatherAUS.csv')


# # Summary of data

# In[67]:


data


# In[68]:


data.shape


# In[69]:


data.describe()


# # Sorting dataset

# In[70]:


X= data.iloc[:,[1,2,3,4,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]].values
Y = data.iloc[:,-1].values


# In[71]:


Y = Y.reshape(-1,1)


# # Dealing with Invalid data

# In[72]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)


# In[46]:


X = imputer.fit_transform(X)
Y = imputer.fit_transform(Y)


# In[47]:


print(X)


# In[48]:


print(Y)


# # Encoding Data

# In[73]:


from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
X[:,0] = le1.fit_transform(X[:,0])
le2 = LabelEncoder()
X[:,4] = le2.fit_transform(X[:,4])
le3 = LabelEncoder()
X[:,6] = le3.fit_transform(X[:,6])
le4 = LabelEncoder()
X[:,7] = le4.fit_transform(X[:,7])
le5 = LabelEncoder()
X[:,-1] = le5.fit_transform(X[:,-1])
le6 = LabelEncoder()
Y[:,-1] = le6.fit_transform(Y[:,-1])


# In[74]:


print(X)


# In[75]:


print(Y)


# In[76]:


Y = np.array(Y,dtype=float)
print(Y)


# #  Feature Scaling

# In[77]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)


# In[78]:


print(X)


# # Changing Dataset into Train and Test

# In[79]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size= 0.2, random_state= 0)


# In[80]:


print(X)


# In[81]:


print(Y)


# # Training Model

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100,random_state=0)
classifier.fit(X_train,Y_train)


# In[ ]:


classifier.score(X_train,Y_train)


# In[ ]:


y_pred = le6.inverse_transform(np.array(classifier.predict(X_test),dtype=int))
Y_test = le6.inverse_transform(np.array(Y_test,dtype=int))


# In[ ]:


print(y_pred)


# In[ ]:


print(Y_test)


# In[ ]:


y_pred = y_pred.reshape(-1,1)
Y_test = Y_test.reshape(-1,1)


# In[ ]:


df = np.concatenate((Y_test,y_pred),axis=1)
dataframe = pd.DataFrame(df,columns=['Rain on Tommorrow','Predition of Rain'])


# In[ ]:


print(dataframe)


# # Calculating Accuracy

# In[66]:


from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred)


# In[ ]:




