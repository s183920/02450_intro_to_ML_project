#1.3 Evaluate the quality of the clustering in terms of your label information for the GMM as well as for 
# the hierarchical clustering where the cut-off is set at the same number of clusters as estimated by the GMM.

from matplotlib.pyplot import figure, show, title, plot, legend
import numpy as np
from scipy.io import loadmat
from __init__ import clusterplot, clusterval
from sklearn.mixture import GaussianMixture
from clean_data import *

# LOAD AND TRANSFORM DATA

data1 = clean_data('Datasets/**videos.csv')
data = transform_data(data1,['likes','dislikes','views','comment_count','trending_time'])
index = [data1.category_id[id] in [10,15,20,22] for id in range(len(data1.category_id))]
data['class'] = data1['category_id']
data = data[index]
data = data.head(200) #first n rows
#print(len(data))
#data = data.sample(100000)
#print(data1)
X = np.array(data[['likes','dislikes','views','comment_count','trending_time']])
y = np.array(data['class'])
attributeNames = ['likes','dislikes','views','comment_count','trending_time']
classNames = ["Mu","Ga","PA","PB"]
#classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

# PLOTS OF GMM, HIER & TRUE LABELS

#HIER PLOT
from cluster_hier import * 

#Set maxclust to GMM pred. 
Maxclust = 5
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# GMM PLOT ringe



# EVALUATION BY SIMM MEASURES at K=GMM, 10.1.3
from sklearn.cluster import k_means

# Maximum number of clusters:
K = 10

# Allocate variables:
Rand = np.zeros((K,))
Jaccard = np.zeros((K,))
NMI = np.zeros((K,))

for k in range(K):
    # run K-means clustering:
    #cls = Pycluster.kcluster(X,k+1)[0]
    centroids, cls, inertia = k_means(X,k+1)
    # compute cluster validities:
    Rand[k], Jaccard[k], NMI[k] = clusterval(y,cls)    
        
# Plot results:

figure(2)
title('Cluster validity')
plot(np.arange(K)+1, Rand)
plot(np.arange(K)+1, Jaccard)
plot(np.arange(K)+1, NMI)
legend(['Rand', 'Jaccard', 'NMI'], loc=4)
print('RAND',Rand)
show()

print('Ran Exercise 10.1.3')




