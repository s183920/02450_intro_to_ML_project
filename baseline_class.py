#Baseline er linear regression with no features
#Computes mean of y on the training data, and use this mean value to predict y on the test data

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from clean_data import clean_data, transform_data

# SETUP ------------------------------------------------------------------------------------------------------
cols = ["likes", "dislikes", "views", "comment_count", "trending_time"]
data = clean_data('Datasets/**videos.csv')
data_norm = transform_data(data, cols)
#Data
X = np.array(data_norm[["likes", "dislikes", "comment_count", "trending_time"]])
y = np.array(data_norm["views"]).squeeze()
attributeNames = ["likes", "dislikes", "comment_count", "trending_time"]
N, M = X.shape


#Split into training and test set. Evt set test- or traing_size. 
X_train, X_test, y_train, y_test = train_test_split(X,y)

#predict fast trending time. Binary classification (fast (< 3) and slow).
#Baseline computes the largest class on the training data, and predict everything in the test-data 
# as belonging to that class (corresponding to the optimal prediction by a logistic regression model with a bias term and no features)

data["class"] = np.where(data["trending_time"]<=3., 1, 0.)

countFast = np.count_nonzero(data["class"] == 1)

baseline_class = (countFast/N)

print(baseline_class)





