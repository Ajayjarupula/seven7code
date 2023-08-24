#!/usr/bin/env python
# coding: utf-8

# # Titanic Classification

# # Importing Libraries

# In[2]:


import numpy as np     
import pandas as pd    
import matplotlib.pyplot as plt     
import seaborn as sns  
from sklearn.linear_model import LogisticRegression     
from sklearn.model_selection import train_test_split    
from sklearn.metrics import confusion_matrix   
from sklearn.metrics import classification_report   
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.tree import DecisionTreeClassifier    
from sklearn.ensemble import RandomForestClassifier    
from sklearn.naive_bayes import GaussianNB    
from sklearn.svm import SVC    


# # Loading dataset

# In[3]:


data = pd.read_csv("C:/Users/admin/Downloads/Titanic-Dataset.csv")
data.head(5)


# In[4]:


data.tail(5)


# In[5]:


print("Shape of the data :- ",data.shape)


# In[6]:


data.info


# In[7]:


data.columns


# In[8]:


data.describe(include="all")


# In[9]:


data.isnull().sum()


# In[10]:


missing_values = data.isna().sum().sort_values(ascending=False)
missing_values


# In[11]:


plt.figure(figsize=(10,5))
missing_values[missing_values != 0].plot.bar()


# In[12]:


def count_plot(feature):
    sns.countplot(x=feature, data=data)
    plt.show()
    print("\n")


# In[13]:


columns = ['Survived','Pclass','Sex','SibSp','Embarked']


# In[14]:


for i in columns:
    count_plot(i)


# In[15]:


data["Age"].plot(kind='hist')


# # Data Pre-Processing

# In[16]:


data.head(5)


# In[17]:


data.drop(['PassengerId', 'Name', 'Cabin', 'Ticket'], axis=1, inplace=True)


# In[18]:


data.head(5)


# In[19]:


data['Age'].fillna(data['Age'].mean(), inplace=True)


# In[20]:


data.isnull().sum()


# In[21]:


data.dtypes


# In[22]:


sex = pd.get_dummies(data["Sex"], drop_first=True)
sex.head()


# In[23]:


embark = pd.get_dummies(data["Embarked"], drop_first=True)
embark.head()


# In[24]:


pclass = pd.get_dummies(data["Pclass"], drop_first=True)
pclass.head()


# In[25]:


data.head(3)


# In[26]:


data.drop(["Sex", "Embarked", "Pclass"], axis=1, inplace=True)


# In[27]:


data.head(1)


# In[28]:


data = pd.concat([data, sex, embark, pclass], axis=1)
data.columns = data.columns.astype(str)
data.head()


# In[29]:


data.columns


# In[31]:


data.dtypes


# # Training Model

# In[32]:


X = data.drop("Survived", axis=1)
Y = data["Survived"]


# In[33]:


Y.shape


# In[34]:


np.unique(Y)


# In[36]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


# # Training Multiple Classification Models

# # Logistic Regression

# In[37]:


log_reg = LogisticRegression(max_iter=1000, C=0.1)
log_reg.fit(X_train, y_train)


# In[38]:


log_reg.score(X_test, y_test)


# In[39]:


log_reg.score(X_train, y_train)


# In[40]:


y_predict = log_reg.predict(X_test)
confusion_matrix(y_test, y_predict)


# In[41]:


classification_report(y_test, y_predict)


# # KNN Classifier

# In[42]:


knn = KNeighborsClassifier(n_neighbors = 13)
knn.fit(X_train,y_train)


# In[43]:


knn.score(X_test,y_test)


# In[44]:


knn.score(X_train,y_train)


# In[45]:


y_predict = knn.predict(X_test)
confusion_matrix(y_test,y_predict)


# In[46]:


print(classification_report(y_test,y_predict))


# # Decision Tree Classifier

# In[47]:


decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)


# In[48]:


decision_tree.score(X_train,y_train)


# In[50]:


decision_tree.score(X_test,y_test)


# In[51]:


y_predict = decision_tree.predict(X_test)
confusion_matrix(y_test,y_predict)


# In[52]:


print(classification_report(y_test,y_predict))


# # Random Forest Classifier

# In[53]:


random_forest = RandomForestClassifier(n_estimators=13)
random_forest.fit(X_train,y_train)


# In[54]:


random_forest.score(X_train,y_train)


# In[55]:


random_forest.score(X_test,y_test)


# In[56]:


y_predict = random_forest.predict(X_test)
confusion_matrix(y_test,y_predict)


# In[57]:


print(classification_report(y_test,y_predict))


# # Gaussian Navie Bayes Classifier

# In[59]:


naive_bayes = GaussianNB()
naive_bayes.fit(X_train,y_train)


# In[60]:


naive_bayes.score(X_train,y_train)


# In[61]:


naive_bayes.score(X_test,y_test)


# In[62]:


y_predict = naive_bayes.predict(X_test)
confusion_matrix(y_test,y_predict)


# In[63]:


print(classification_report(y_test,y_predict))


# # SVM Classifier

# In[64]:


svm = SVC()
svm.fit(X_train,y_train)


# In[65]:


svm.score(X_test,y_test)


# In[66]:


svm.score(X_train,y_train)


# In[67]:


y_predict = svm.predict(X_test)
confusion_matrix(y_test,y_predict)


# In[68]:


print(classification_report(y_test,y_predict))


# In[ ]:




