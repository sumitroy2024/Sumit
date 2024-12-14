#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[28]:


dataset = pd.read_csv("NewWine.csv")


# In[29]:


dataset


# In[30]:


X = dataset.iloc[:,:-1].values


# In[31]:


y = dataset.iloc[:,-1].values


# In[32]:


X


# In[33]:


y


# In[34]:


from sklearn.model_selection import train_test_split


# In[35]:


X_train , X_test, y_train , y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# In[36]:


y_test


# In[37]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


# In[38]:


X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[39]:


X_train


# In[40]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train = lda.fit_transform(X_train , y_train)
X_test = lda.transform(X_test)


# In[41]:


X_train


# In[45]:


from sklearn.linear_model import LogisticRegression as LR
lr = LR()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)


# In[46]:


y_test


# In[47]:


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[48]:


from sklearn.metrics import accuracy_score


# In[49]:


accuracy_score(y_test, y_pred)


# In[50]:


import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


# In[52]:


X_set, y_set = X_train , y_train
X1, X2 = np.meshgrid(
            np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.25),
            np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.25)
                    )
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red','green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1],
               c = ListedColormap(('red','green','blue'))(i), label = j)


# In[53]:


X_set, y_set = X_test , y_test
X1, X2 = np.meshgrid(
            np.arange(start = X_set[:,0].min()-1, stop = X_set[:,0].max()+1, step = 0.25),
            np.arange(start = X_set[:,1].min()-1, stop = X_set[:,1].max()+1, step = 0.25)
                    )
plt.contourf(X1, X2, lr.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
            alpha = 0.75, cmap = ListedColormap(('red','green','blue')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1],
               c = ListedColormap(('red','green','blue'))(i), label = j)


# In[ ]:




