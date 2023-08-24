#!/usr/bin/env python
# coding: utf-8

# # Mall Customer Segmentation

# # Importing libraries

# In[43]:


import pandas as pd
import numpy as np
from numpy import unique
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer,intercluster_distance
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth
from scipy.cluster.hierarchy import dendrogram, linkage

from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import pairwise_distances
from sklearn import metrics

import warnings
warnings.filterwarnings("ignore")


# # Loading dataset

# In[8]:


df=pd.read_csv('C:/Users/admin/Downloads/Mall_Customers.csv')
df


# In[9]:


df.describe(include='all')


# In[10]:


df.dtypes


# In[11]:


df.isnull().sum()


# # Data  Visualization

# # Distribution Plot

# In[12]:


plt.figure(figsize=(11,8))
sns.displot(data=df[[ 'Age', 'Annual Income (k$)',
       'Spending Score (1-100)']],multiple='stack',kde=True,palette='magma',rug=True,stat='probability')
plt.title('Distribution Plot of Age, Annual Income, Spending Score')
plt.show()


# In[13]:


fig,axes = plt.subplots(1, 2,figsize=(10,6))
sns.countplot(y=df['Gender'],alpha=0.85,ax=axes[0],palette='tab10')
sns.barplot(data=df,y='Age',x='Gender',hue='Gender',errorbar=('ci',40),
            palette='Accent',errcolor="purple",errwidth=2,capsize=0.4,ax=axes[1])
sns.move_legend(axes[1], "upper left",bbox_to_anchor=(1, 1))
plt.show()


# In[14]:


plt.figure(figsize=(15,9))
ax=sns.pairplot(data=df,vars=['Age','Annual Income (k$)','Spending Score (1-100)'],hue='Gender',
             palette='Set1',diag_kind='hist',kind='scatter')
ax.map_upper(sns.regplot)
sns.move_legend(ax, "upper left",bbox_to_anchor=(1, 1))
plt.show()


# In[15]:


plt.figure(figsize=(7, 5))
df1=df[['Age','Annual Income (k$)','Spending Score (1-100)']]
sns.heatmap(df1.corr(), annot=True,linewidth=.8, cmap="Reds")


# In[16]:


plt.figure(figsize=(9,5))
sns.scatterplot(df,x='Annual Income (k$)',y='Spending Score (1-100)',hue='Gender',palette='seismic',s=100,alpha=0.7)
plt.title('Annual Income vs Spending Score')
plt.show()


# In[17]:


plt.figure(figsize=(9,5))
sns.scatterplot(df,x='Age',y='Spending Score (1-100)',hue='Gender',palette='CMRmap',s=100,alpha=0.7)

plt.title('Age vs Spending Score')
plt.show()


# In[18]:


X1=df[['Age','Annual Income (k$)','Spending Score (1-100)']]


# # 1. KMeans

# In[19]:


visualizer = KElbowVisualizer(KMeans(),k=(2,10))
visualizer.fit(X1) 
visualizer.poof()
plt.show()


# In[20]:


visualizer = KElbowVisualizer(KMeans(random_state=23),k=(2,10),metric='silhouette')
visualizer.fit(X1) 
visualizer.show()
plt.show()


# In[21]:


sil_visualizer = SilhouetteVisualizer(KMeans(5))
sil_visualizer.fit(X1)
sil_visualizer.show()
plt.show()


# In[22]:


sil_visualizer = SilhouetteVisualizer(KMeans(4))
sil_visualizer.fit(X1)
sil_visualizer.show()
plt.show()


# In[23]:


#Now we know that optimal k value is 5
k=5
K_means=KMeans(init="k-means++",n_clusters=k)
K_means.fit(X1)


# In[24]:


df['KM_Cluster']=K_means.labels_
KM_centres=K_means.cluster_centers_


# In[25]:


plt.figure(figsize=(9,5))
ax=sns.scatterplot(data=df,x='Annual Income (k$)', y='Spending Score (1-100)',hue='KM_Cluster',palette='tab10',s=90)
ax=sns.scatterplot(x=KM_centres[:,1],y=KM_centres[:,2],s=120,color='black')
sns.move_legend(ax, "upper left",bbox_to_anchor=(1, 1))
plt.show()


# In[26]:


plt.figure(figsize=(9,5))
ax=sns.scatterplot(data=df,x='Age', y='Spending Score (1-100)',hue='KM_Cluster',palette='plasma',s=90)
ax=sns.scatterplot(x=KM_centres[:,0],y=KM_centres[:,2],s=120,color='black')
sns.move_legend(ax, "upper left",bbox_to_anchor=(1, 1))
plt.show()


# # 2. Heirarchical Clustering

# In[27]:


plt.figure(figsize=(11, 6))
dendo = dendrogram(linkage(X1, method='ward'),leaf_font_size=5,truncate_mode = 'lastp') 
plt.axhline(y=200 , color='black',linestyle = '--')
plt.title('Dendrogram', fontsize=25) 
plt.xlabel('Customer Counts')
plt.ylabel('Euclidian Distances')
plt.yticks(fontsize=13)  
plt.xticks(fontsize=13) 
plt.show()  


# In[28]:


agc = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
agc.fit(X1)
df['HC_Cluster']=agc.labels_


# In[29]:


plt.figure(figsize=(9,5))
ax=sns.scatterplot(data=df,x='Annual Income (k$)', y='Spending Score (1-100)',hue='HC_Cluster',palette='tab10',s=90)
sns.move_legend(ax, "upper left",bbox_to_anchor=(1, 1))
plt.show()


# # 3. DBSCAN

# In[30]:


db = DBSCAN(eps=10,min_samples=6) #arbitrary eps and min_sample values
db.fit(X1) 
tlabels = db.labels_
tlabels


# In[31]:


ep,count = NearestNeighbors(n_neighbors=20).fit(X1).kneighbors(X1) #arbitrary n_neighbors value
ep = np.sort(ep, axis=0)
plt.figure(figsize=(10,8))
plt.plot(ep[:,1])
plt.show()


# In[32]:


min_samples = range(2,10)
eps = np.arange(9,13, 0.25)
output = []
for ms in min_samples:
    for ep in eps:
        labels = DBSCAN(eps = ep,min_samples=ms).fit(X1).labels_
        score = metrics.silhouette_score(X1, labels)
        output.append((ms, ep, score))
min_samples, eps, score = sorted(output, key=lambda x:x[-1])[-1]
print(f"Best silhouette_score: {score}")
print(f"min_samples: {min_samples}")
print(f"eps: {eps}")      


# In[33]:


db = DBSCAN(eps=12.5,min_samples=4)
db.fit(X1) 
labels = db.labels_
df['DB_Cluster']=labels
labels


# In[34]:


plt.figure(figsize=(9,5))
ax=sns.scatterplot(data=df,x='Annual Income (k$)', y='Spending Score (1-100)',hue='DB_Cluster',palette='Set1',s=90)
# ax=sns.scatterplot(x=centres[:,0],y=centres[:,2],s=120,color='black')
sns.move_legend(ax, "upper left",bbox_to_anchor=(1, 1))
plt.show()


# # 4. Mean Shift

# In[35]:


estimate_bandwidth(X1, quantile=0.1)


# In[36]:


mean_shift = MeanShift(bandwidth=22.173844534734847)
mean_shift.fit(X1)


# In[37]:


yhat_ms = mean_shift.predict(X1)
clusters_ms = unique(yhat_ms)
print("Clusters of Mean Shift:", clusters_ms)


# In[38]:


labels_ms =mean_shift.labels_ 
df['MS_Cluster']=labels_ms
centroids_ms = mean_shift.cluster_centers_


# In[39]:


plt.figure(figsize=(9,5))
ax=sns.scatterplot(data=df,x='Annual Income (k$)', y='Spending Score (1-100)',hue='MS_Cluster',palette='Set1',s=90)
ax=sns.scatterplot(x=centroids_ms[:,1],y=centroids_ms[:,2],s=120,color='black')
sns.move_legend(ax, "upper left",bbox_to_anchor=(1, 1))
plt.show()


# Model Evaluation metrics

# In[40]:


silhoutte_scores=[metrics.silhouette_score(X1,K_means.labels_),metrics.silhouette_score(X1,agc.labels_),
                  metrics.silhouette_score(X1,db.labels_),metrics.silhouette_score(X1,mean_shift.labels_)]

davies_bouldin_scores=[metrics.davies_bouldin_score(X1,K_means.labels_),metrics.davies_bouldin_score(X1,agc.labels_),
                  metrics.davies_bouldin_score(X1,db.labels_),metrics.davies_bouldin_score(X1,mean_shift.labels_)]


# In[41]:


score_df={'Algorithm':["K-means", "Heirarchical", "DBSCAN", "Mean-Shift"],'Silhouette Score':silhoutte_scores,
          'Davies-Bouldin Scores':davies_bouldin_scores}
score_df=pd.DataFrame.from_dict(score_df)
score_df


# In[42]:


fig,axes = plt.subplots(1, 2, figsize=(10,5))
sns.lineplot(data=score_df,x='Algorithm',y='Silhouette Score',color='Magenta',ax=axes[0])
sns.lineplot(data=score_df,x='Algorithm',y='Davies-Bouldin Scores',ax=axes[1])
plt.show()


# In[ ]:




