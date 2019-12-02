#1.1 report 3 

from matplotlib.pyplot import figure, show, legend
from scipy.io import loadmat
from __init__ import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from clean_data import *


# LOAD AND TRANSFORM DATA

data1 = clean_data('Datasets/**videos.csv')
data = transform_data(data1,['likes','dislikes','views','comment_count','trending_time'])
index = [data1.category_id[id] in [10,15,20,22] for id in range(len(data1.category_id))]
data['class'] = data1['category_id']
data = data[index]
data = data.head(2000) #first n rows
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

#ex 10.1.2
#Use linkage to create a sample to sample distance matrix according to given distance metric 
# and creates the linkages  between  data  points  forming  the  hierarchical  cluster  tree


# Perform hierarchical/agglomerative clustering on data matrix
# Change these to 
Method = 'average'
Metric = 'correlation'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
# Select appropiate dissimilarity measure 

from categories import find_cat_labels

cat_labels = find_cat_labels('Datasets/CA_category_id.json')
classLabels = [cat_labels[i] for i in ["10","15","20","22"]] 

Maxclust = 12
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
legend(classLabels)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
# Cutof dendogram at threshold
max_display_levels=4
figure(2,figsize=(10,5))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()

print('Ran Exercise 10.2.1')
