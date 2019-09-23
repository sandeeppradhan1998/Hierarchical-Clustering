# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 02:29:57 2019

@author: Dilip
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#import dataset 
dataframe=pd.read_csv('Customers.csv')
x=dataframe.iloc[:,[3,4]].values


# Using the dendrogram 
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

# Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)

#visualize the cluster
plt.scatter(x[y_hc==0,0], x[y_hc==0,1], s=150, c='blue', label='Cluster1')
plt.scatter(x[y_hc==1,0], x[y_hc==1,1], s=150, c='orange', label='Cluster2')
plt.scatter(x[y_hc==2,0], x[y_hc==2,1], s=150, c='green', label='Cluster3')


plt.title('Clusters of customers')
plt.xlabel('Annual Income(K$)')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()






    
