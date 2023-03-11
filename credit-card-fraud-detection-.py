#!/usr/bin/env python
# coding: utf-8

# ## Basic Pakages

# In[1]:


import numpy as np 
import pandas as pd


# ## Import Dataset

# In[2]:


fraudTrain = pd.read_csv("fraudTrain.csv")
fraudTest = pd.read_csv("fraudTest.csv")


# ## DataSet Properties

# In[3]:


fraudTrain.head()


# In[68]:


fraudTrain.describe()


# In[69]:


fraudTrain.shape


# In[71]:


print("No of Fraud",len(fraudTrain[fraudTrain['is_fraud']==1]))


# In[4]:


fraudTest.head()


# In[65]:


fraudTrain.describe()


# In[66]:


fraudTrain.shape


# In[ ]:


print("No of Fraud",len(fraudTrain[fraudTrain['is_fraud']==1]))


# In[5]:


fraudTrain.drop("Unnamed: 0",axis=1,inplace=True) 
fraudTest.drop("Unnamed: 0",axis=1,inplace=True) 
fraudTrain = fraudTrain.drop(['cc_num','first','last','trans_num'],axis=1)
fraudTest = fraudTest.drop(['cc_num','first','last','trans_num'],axis=1)


# In[6]:


from datetime import datetime as dt
fraudTrain["trans_date_trans_time"] = pd.to_datetime(fraudTrain["trans_date_trans_time"])
fraudTrain["trans_date"] = fraudTrain["trans_date_trans_time"].dt.date
fraudTrain["trans_date"]= pd.to_datetime(fraudTrain["trans_date"])

fraudTrain['year'] = fraudTrain['trans_date'].dt.year
fraudTrain['month'] = fraudTrain['trans_date'].dt.month
fraudTrain['day'] = fraudTrain['trans_date'].dt.day

fraudTest["trans_date_trans_time"] = pd.to_datetime(fraudTest["trans_date_trans_time"])
fraudTest["trans_date"] = fraudTest["trans_date_trans_time"].dt.date
fraudTest["trans_date"]= pd.to_datetime(fraudTest["trans_date"])

fraudTest['year'] = fraudTest['trans_date'].dt.year
fraudTest['month'] = fraudTest['trans_date'].dt.month
fraudTest['day'] = fraudTest['trans_date'].dt.day


# **Show the number of fraud / normal transactions**

# In[7]:


fraudTrain["is_fraud"].value_counts()


# **Mean Amount difference between fraud / normal transactions**

# In[8]:


fraudTrain.groupby("is_fraud")['amt'].mean()


# **Show number of frauds by categories**

# In[9]:



pd.crosstab(fraudTrain["category"],fraudTrain["is_fraud"])


# **Show the proportion of fraud by categories**

# In[10]:



pd.crosstab(fraudTrain["category"],fraudTrain["is_fraud"],normalize='index')


# ## Preprocessing

# In[11]:


fraudTrain = pd.get_dummies(fraudTrain,columns=['category'],drop_first=True)
fraudTest = pd.get_dummies(fraudTest,columns=['category'],drop_first=True)

fraudTrain.columns = fraudTrain.columns.str.replace(' ', '')
fraudTest.columns = fraudTest.columns.str.replace(' ', '')

train = fraudTrain.select_dtypes(include='number')
test = fraudTest.select_dtypes(include='number')


# In[12]:


total = pd.concat([train, test])


# In[13]:


X = total.drop("is_fraud",axis=1) 
y = total["is_fraud"]


# In[14]:


X = X.drop(['zip','lat','long','unix_time','merch_lat','merch_long'],axis=1)


# ## Split

# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.3, random_state=42)


# ## Metrices

# In[16]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# # DecisionTreeClassifier

# In[17]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)


# In[18]:


from sklearn.metrics import accuracy_score, mean_absolute_error ,mean_squared_error, confusion_matrix, median_absolute_error,classification_report, f1_score,recall_score,precision_score

print("Score the X-train with Y-train is : ", dtc.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", dtc.score(X_test,y_test))

y_pred=dtc.predict(X_test)

print("Accuracy score " , accuracy_score(y_test,y_pred))
print("F1 score: ", round(f1_score(y_test, y_pred, average='weighted')*100,2),"%")


# In[19]:


matrix = classification_report(y_test,y_pred,labels=[1,0])
print('Classification report : \n',matrix)


# In[20]:


print('Precision: %.3f' % precision_score(y_test, y_pred))


# In[21]:


print("F1 score: ", round(f1_score(y_test, y_pred, average='weighted')*100,2),"%")


# ## RandomForestClassifier

# In[22]:


from sklearn.ensemble import RandomForestClassifier


# In[23]:


rfc = RandomForestClassifier()


# In[24]:


rfc.fit(X_train,y_train)


# In[37]:


print("Score the X-train with Y-train is : ", rfc.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", rfc.score(X_test,y_test))

y_pred=rfc.predict(X_test)

print("Accuracy score " , accuracy_score(y_test,y_pred))
print("F1 score: ", round(f1_score(y_test, y_pred, average='weighted')*100,2),"%")


# In[38]:


matrix = classification_report(y_test,y_pred,labels=[1,0])
print('Classification report : \n',matrix)


# In[39]:


print('Precision: %.3f' % precision_score(y_test, y_pred))


# In[40]:


print("F1 score: ", round(f1_score(y_test, y_pred, average='weighted')*100,2),"%")


# ## Support Vector Machine (SVM)

# In[41]:


from sklearn import svm


# In[42]:


SVM = svm.SVC(kernel='linear')


# In[43]:


SVM.fit(X_train,y_train)


# In[44]:


print("Score the X-train with Y-train is : ", SVM.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", SVM.score(X_test,y_test))

y_pred=SVM.predict(X_test)

print("Accuracy score " , accuracy_score(y_test,y_pred))
print("F1 score: ", round(f1_score(y_test, y_pred, average='weighted')*100,2),"%")


# In[57]:


import warnings
warnings.filterwarnings("ignore")


# In[58]:


matrix = classification_report(y_test,y_pred,labels=[1,0])
print('Classification report : \n',matrix)


# In[59]:


print('Precision: %.3f' % precision_score(y_test, y_pred))


# In[60]:


print("F1 score: ", round(f1_score(y_test, y_pred, average='weighted')*100,2),"%")


# ## KNeighborsClassifier

# In[61]:


from sklearn.neighbors import KNeighborsClassifier


# In[62]:


knn = KNeighborsClassifier(n_neighbors=3)


# In[63]:


knn.fit(X_train,y_train)


# In[ ]:


print("Score the X-train with Y-train is : ", knn.score(X_train,y_train))
print("Score the X-test  with Y-test  is : ", knn.score(X_test,y_test))

y_pred=knn.predict(X_test)

print("Accuracy score " , accuracy_score(y_test,y_pred))
print("F1 score: ", round(f1_score(y_test, y_pred, average='weighted')*100,2),"%")


# In[ ]:


matrix = classification_report(y_test,y_pred,labels=[1,0])
print('Classification report : \n',matrix)


# In[ ]:


print('Precision: %.3f' % precision_score(y_test, y_pred))


# In[ ]:


print("F1 score: ", round(f1_score(y_test, y_pred, average='weighted')*100,2),"%")

