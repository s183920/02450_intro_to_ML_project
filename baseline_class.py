#Baseline er linear regression with no features
#Computes mean of y on the training data, and use this mean value to predict y on the test data

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from clean_data import clean_data, transform_data
import matplotlib.pyplot as plt


#Data
#X = np.array(data_norm[["likes", "dislikes", "comment_count", "trending_time"]])
#y = np.array(data_norm["views"]).squeeze()
#attributeNames = ["likes", "dislikes", "comment_count", "trending_time"]
#N, M = X.shape

# SETUP ------------------------------------------------------------------------------------------------------
cols = ["likes", "dislikes", "views", "comment_count", "trending_time"]
data = clean_data('Datasets/**videos.csv')
data = data.head(1000)
data_norm = transform_data(data, cols)

data["class"] = np.where(data["trending_time"]<=3., 1, 0.)
#Data
X = np.array(data_norm[["likes", "dislikes", "views", "comment_count", "trending_time"]])
y = np.array(data["class"]).squeeze()
print(y)
attributeNames = ["likes", "dislikes", "views", "comment_count", "trending_time"]
N, M = X.shape


#Split into training and test set. Evt set test- or traing_size. 
K = 10
CV = model_selection.KFold(K, shuffle=True)

errGen = np.zeros((N,K))
k=0
for train_index, test_index in CV.split(X,y):
    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    #extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
	
	countFast1 = np.count_nonzero(y_train == 1) 
	baseline_class1 = (countFast/y_train.size) 
	countFast2 = np.count_nonzero(y_test == 1) 
	baseline_class2 = (countFast2/y_test.size)

	error_test = ((baseline_class2 / baseline_class1)-1)*100

	errGen[train_index]+=error_test

	k+=1


errors = 100*sum(errGen,0)/N


#sorted list of index
res = errors
sort_index = np.argsort(res)

errors = 100*sum(errors,0)/N
print("Error percent ", errGen)
print("sort_index", sort_index)





#X_train, X_test, y_train, y_test = train_test_split(X,y)

#predict fast trending time. Binary classification (fast (< 3) and slow).
#Baseline computes the largest class on the training data, and predict everything in the test-data 
# as belonging to that class (corresponding to the optimal prediction by a logistic regression model with a bias term and no features)

#data["class"] = np.where(data["trending_time"]<=3., 1, 0.)

#countFast = np.count_nonzero(data["class"] == 1)

#baseline_class = (countFast/N)

#print(baseline_class)
baseline_class = (countFast/N)
print(countFast)
print(baseline_class)





