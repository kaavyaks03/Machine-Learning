#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


#Read the CSV file
dataset=pd.read_csv("Iris dataset .csv")


# In[3]:


#assigning x & y columns
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
print(x)
print(y)


# In[4]:


#Train, test split of the dataset
from sklearn.model_selection import train_test_split


# In[5]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[6]:


#Feature Scaling
from sklearn.preprocessing import StandardScaler


# In[7]:


s=StandardScaler()
x_train=s.fit_transform(x_train)
x_test=s.transform(x_test)


# In[8]:


#fitting the model using KNN algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[10]:


classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)


# In[11]:


#Testing the model with test data
y_pred=classifier.predict(x_test)
print(y_pred)


# In[12]:


#testing the model with train data
y_ans=classifier.predict(x_train)
print(y_ans)


# In[13]:


#accuracy score and confusion matrix of the model
from sklearn.metrics import confusion_matrix,accuracy_score
a1=accuracy_score(y_train,y_ans)
print(a1)
a2=accuracy_score(y_test,y_pred)
print(a2)


# In[14]:


cm1=confusion_matrix(y_train,y_ans)
print(cm1)
cm2=confusion_matrix(y_test,y_pred)
print(cm2)


# In[15]:


#visualization of the result
import matplotlib.pyplot as plt
plt.plot(y_test,y_pred,'o')
plt.title("Iris Classificatiion using KNN")
plt.xlabel("Predicted Result")
plt.ylabel("Actual Result")
plt.show()

