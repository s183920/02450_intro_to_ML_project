# 1.2 report 3
#ex 11.1.1

from matplotlib.pyplot import figure, show, legend, plot, xlabel
import numpy as np
from scipy.io import loadmat
from __init__ import clusterplot
from sklearn.mixture import GaussianMixture
from clean_data import *

# LOAD AND TRANSFORM DATA

data1 = clean_data('Datasets/**videos.csv')
data = transform_data(data1,['likes','dislikes','views','comment_count','trending_time'])
index = [data1.category_id[id] in [10,15,20,22] for id in range(len(data1.category_id))]
data['class'] = data1['category_id']
data = data[index]
data =data.head(5000) #first n rows
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

# Number of clusters
K = 4
cov_type = 'full' # e.g. 'full' or 'diag'


# define the initialization procedure (initial value of means)
initialization_method = 'random'#  'random' or 'kmeans'
# random signifies random initiation, kmeans means we run a K-means and use the
# result as the starting point. K-means might converge faster/better than  
# random, but might also cause the algorithm to be stuck in a poor local minimum 


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
from categories import find_cat_labels

cat_labels = find_cat_labels('Datasets/CA_category_id.json')
classLabels = [cat_labels[i] for i in ["10","15","20","22"]] 

#figure(figsize=(14,9))
legend(classLabels)
#clusterplot(X, clusterid=cls, centroids=cds, y=y, covars=covs)
#show()

## In case the number of features != 2, then a subset of features most be plotted instead.
figure(figsize=(14,9))
idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
show()

## In case the number of features != 2, then a subset of features most be plotted instead.
#figure(figsize=(14,9))
#idx = [0,1] # feature index, choose two features to use as x and y axis in the plot
#clusterplot(X[:,idx], clusterid=cls, centroids=cds[:,idx], y=y, covars=covs[:,idx,:][:,:,idx])
#show()

print('Ran Exercise 11.1.1')



#from ex 11.1.5:

from sklearn import model_selection


# Range of K's to try
KRange = range(1,15)
T = len(KRange)

covar_type = 'full'       # you can try out 'diag' as well
reps = 3                  # number of fits with different initalizations, best result will be kept
init_procedure = 'kmeans' # 'kmeans' or 'random'

# Allocate variables
#BIC = np.zeros((T,))
#AIC = np.zeros((T,))
CVE = np.zeros((T,))

# K-fold crossvalidation
CV = model_selection.KFold(n_splits=10,shuffle=True)

for t,K in enumerate(KRange):
        print('Fitting model for K={0}'.format(K))

        # Fit Gaussian mixture model
        gmm = GaussianMixture(n_components=K, covariance_type=covar_type, 
                              n_init=reps, init_params=init_procedure,
                              tol=1e-6, reg_covar=1e-6).fit(X)
        
        # Get BIC and AIC
        #BIC[t,] = gmm.bic(X)
        #AIC[t,] = gmm.aic(X)

        # For each crossvalidation fold
        for train_index, test_index in CV.split(X):

            # extract training and test set for current CV fold
            X_train = X[train_index]
            X_test = X[test_index]

            # Fit Gaussian mixture model to X_train
            gmm = GaussianMixture(n_components=K, covariance_type=covar_type, n_init=reps).fit(X_train)

            # compute negative log likelihood of X_test
            CVE[t] += -gmm.score_samples(X_test).sum()
            

# Plot results



#Print K multivGaussians to model the data
#To evaluate the model fit on the test set in terms of negative log likelihood
minCVE = min(CVE)
KMIN = np.nonzero(CVE == minCVE)[0][0] +1
print("Best K", KMIN)


figure(1); 
plot(KRange, 2*CVE,'-ok')
legend('Crossvalidation')
xlabel('K')
show()

print('Ran Exercise 11.1.5')

#Look at plot and determine for which number of K is best