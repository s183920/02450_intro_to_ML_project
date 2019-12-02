# a model for classification(predicting a binary class label withlogisticregression) using the above regulariza-tion method 
# (also called ridge regression, or L2-regularization).
#When regularizing a model we apply the same regularization strength to all features, and so it is important 
# to standardize the features. 

# From exercise 8.1.2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from clean_data import clean_data, transform_data

import sklearn.linear_model as lm
from sklearn import model_selection
from __init__ import feature_selector_lr, bmplot


# SETUP ------------------------------------------------------------------------------------------------------
cols = ["likes", "dislikes", "views", "comment_count", "trending_time"]
data = clean_data('Datasets/**videos.csv')
data_norm = transform_data(data, cols)

data["class"] = np.where(data["trending_time"]<=3., 1, 0.)
#Data
X = np.array(data_norm[["likes", "dislikes", "views", "comment_count", "trending_time"]])
y = np.array(data["class"]).squeeze()
attributeNames = ["likes", "dislikes", "views", "comment_count", "trending_time"]
N, M = X.shape

# Create crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
K = 10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
# Try to change the test_size to e.g. 50 % and 99 % - how does that change the 
# effect of regularization? 

font_size = 15
plt.rcParams.update({'font.size': font_size})

# Standardize the training and set set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

#LOGISTIC REGRESSION MODEL

# Fit regularized logistic regression model to training data to predict 
lambda_interval = np.logspace(-10, 15, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
for k in range(0, len(lambda_interval)):
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k])
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate)
opt_lambda = lambda_interval[opt_lambda_idx]
max_error = np.max(test_error_rate)
opt_lambda_idxM = np.argmax(test_error_rate)
opt_lambdaM = lambda_interval[opt_lambda_idxM]
print("TTEEEEEESSSSSTTTTT",opt_lambdaM)

plt.figure(figsize=(8,8))
#plt.plot(np.log10(lambda_interval), train_error_rate*100)
#plt.plot(np.log10(lambda_interval), test_error_rate*100)
#plt.plot(np.log10(opt_lambda), min_error*100, 'o')
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 3, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2), "maxT"+str(np.round(np.log10(opt_lambdaM),2))))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 4])
plt.grid()
plt.show()    

plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    

print('Ran Regress_class')





# A suit-able regularization strength is determined using hold-out cross-validation.

#We train 10 pct
#How is the magnitude of the fitted parameters affected by the regularizations trength?
