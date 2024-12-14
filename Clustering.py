#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


dataset = pd.read_csv("YrCount.csv")
X = dataset.iloc[:, :].values


# In[6]:


X


# In[8]:


import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.title("Dendrogram")
plt.xlabel("Data")
plt.ylabel("Eucledian Distance")
plt.show()


# In[9]:


from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters = 5)
y_hc = clustering.fit_predict(X)


# In[10]:


y_hc


# In[12]:


plt.scatter(X[y_hc == 0 , 0] ,X[y_hc == 0 , 1], c= 'red' , label = 'Cluster1' )
plt.scatter(X[y_hc == 1 , 0] ,X[y_hc == 1 , 1], c= 'green' , label = 'Cluster2' )
plt.scatter(X[y_hc == 2 , 0] ,X[y_hc == 2 , 1], c= 'pink' , label = 'Cluster3' )
plt.scatter(X[y_hc == 3 , 0] ,X[y_hc == 3 , 1], c= 'orange' , label = 'Cluster4' )
plt.scatter(X[y_hc == 4 , 0] ,X[y_hc == 4 , 1], c= 'blue' , label = 'Cluster5' )
plt.title("Cluster")
plt.xlabel("Period")
plt.ylabel("count")
plt.legend()
plt.show()


# 
