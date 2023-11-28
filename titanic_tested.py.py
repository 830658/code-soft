#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 


# In[2]:


df=pd.read_csv('tested.csv')


# In[3]:


df


# In[4]:


print(df.head())


# In[5]:


df.shape


# In[6]:


df.tail()


# In[7]:


df.describe()


# In[8]:


df.isnull().sum().sum()


# In[9]:


df.isnull()


# In[10]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[11]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',data=df)


# In[12]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=df,palette='RdBu_r')


# In[13]:


sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=df,palette='rainbow')


# In[14]:


sns.distplot(df['Age'].dropna(),kde=False,color='darkred',bins=40)


# In[15]:


df['Age'].hist(bins=30,color='darkred',alpha=0.3)


# In[16]:


sns.countplot(x='SibSp',data=df)


# In[17]:


df['Fare'].hist(color='green',bins=40,figsize=(8,4))


# In[18]:


plt.figure(figsize=(12,7))
sns.boxplot(x='Pclass',y='Age',data=df,palette='winter')


# In[19]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        
        if Pclass == 1 :
            return 43
        
        elif Pclass == 2 :
            return 27
        
        else:
            return 24 
        
    else:
        return Age


# In[20]:


df['Age']=df[['Age','Pclass']].apply(impute_age,axis=1)


# In[21]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[22]:


df.drop('Cabin',axis=1,inplace=True)


# In[23]:


df.head()


# In[24]:


df.dropna(inplace=True)


# In[25]:


df.info


# In[26]:


pd.get_dummies(df['Embarked'],drop_first=True).head()


# In[27]:


sex=pd.get_dummies(df['Sex'],drop_first=True)
embark=pd.get_dummies(df['Embarked'],drop_first=True)


# In[28]:


df.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[29]:


df.head()


# In[30]:


df=pd.concat([df,sex,embark],axis=1)


# In[31]:


df.head()


# In[32]:


df.drop('Survived',axis=1).head()


# In[33]:


df['Survived'].head()


# In[34]:


X = df.drop('Survived', axis=1) 
y = df['Survived'] 
print(X.head())
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)


# In[35]:


from sklearn.metrics import r2_score
score = r2_score(y_test,logmodel.predict(X_test))
print(score)

