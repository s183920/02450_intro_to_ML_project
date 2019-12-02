from matplotlib.pyplot import figure, show
from scipy.io import loadmat
from __init__ import clusterplot
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import numpy as np
from clean_data import clean_data, transform_data
from sklearn.mixture import GaussianMixture

#--------------------------IMPORTING DATA--------------------------------
data1 = clean_data('Datasets/**videos.csv')
data = transform_data(data1,['likes','dislikes','views','comment_count','trending_time'])
index = [data1.category_id[id] in [10,15,20,22,26,30] for id in range(len(data1.category_id))]
data['class'] = data1['category_id']
data = data[index]
#print(len(data))
#data = data.sample(100000)
#print(data1)
X = np.array(data[['likes','dislikes','views','comment_count','trending_time']])
y = np.array(data['class'])
attributeNames = ['likes','dislikes','views','comment_count','trending_time']
classNames = ["10","15","20","22","26","30"]

#classNames = [name[0][0] for name in mat_data['classNames']]
N, M = X.shape
C = len(classNames)

'''
# Perform hierarchical/agglomerative clustering on data matrix
Method = 'single'
Metric = 'euclidean'

Z = linkage(X, method=Method, metric=Metric)

# Compute and display clusters by thresholding the dendrogram
Maxclust = 6
cls = fcluster(Z, criterion='maxclust', t=Maxclust)
figure(1)
clusterplot(X, cls.reshape(cls.shape[0],1), y=y)

# Display dendrogram
max_display_levels=6
figure(2,figsize=(10,4))
dendrogram(Z, truncate_mode='level', p=max_display_levels)

show()
'''
#Exercise 11.1
K = 6
cov_type = 'full' # e.g. 'full' or 'diag'

# define the initialization procedure (initial value of means)
initialization_method = 'random'#  'random' or 'kmeans'
# random signifies random initiation, kmeans means we run a K-means and use the
# result as the starting point. K-means might converge faster/better than  
# random, but might also cause the algorithm to be stuck in a poor local minimum 

# type of covariance, you can try out 'diag' as well
reps = 1
# number of fits with different initalizations, best result will be kept
# Fit Gaussian mixture model
gmm = GaussianMixture(n_components=K, covariance_type=cov_type, n_init=reps, 
                      tol=1e-6, reg_covar=1e-6, init_params=initialization_method).fit(X)
cls = gmm.predict(X)    
# extract cluster labels
cds = gmm.means_        
# extract cluster centroids (means of gaussians)
covs = gmm.covariances_
# extract cluster shapes (covariances of gaussians)
if cov_type.lower() == 'diag':
    new_covs = np.zeros([K,M,M])    
    
    count = 0    
    for elem in covs:
        temp_m = np.zeros([M,M])
        new_covs[count] = np.diag(elem)
        count += 1

    covs = new_covs

# Plot results:
#figure(figsize=(14,9))
#clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
#show()

figure(figsize=(14,9))
idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
show()