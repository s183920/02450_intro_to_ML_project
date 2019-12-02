from matplotlib.pylab import figure, plot, xlabel, ylabel, legend, ylim, show
import sklearn.linear_model as lm
from matplotlib.pyplot import figure, boxplot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from scipy.io import loadmat
import torch
from sklearn import model_selection
from __init__ import train_neural_net, draw_neural_net, mcnemar
from scipy import stats
from clean_data import clean_data, transform_data

#-----------------------LOADING DATA----------------------------

data = clean_data('Datasets/**videos.csv')
data = transform_data(data,['likes','dislikes','views','comment_count','trending_time'])
np.random.seed(180820)
#data = data.head(100000)
X = np.array(data[['likes','dislikes','views','comment_count','trending_time']])
#y = np.array(data['views']).squeeze()
data['class'] = np.where(data["trending_time"]<=3., 1, 0.)
y = np.where(data["trending_time"]<=3., 1, 0.)
#X = np.array(data)
#y = X[:,[4]]             
#X = X[:,0:4]
attributeNames = ['likes','dislikes','views','comment_count','trending_time']
N, M = X.shape


#----------------------TAKEN FROM EXERCISE 5.2.6------------------

Logregpredict = []
KNNpredict = []
Baselinepredict = []
CV1 = model_selection.KFold(n_splits=10,shuffle=True)

for train_index_i, test_index_i in CV1.split(X,y):
    # extract training and test set for current CV fold
    X_train = X[train_index_i,:]
    y_train = y[train_index_i]
    X_test = X[test_index_i,:]
    y_test = y[test_index_i]


#LogReg
model = lm.logistic.LogisticRegression(C=40,penalty='l2')
model = model.fit(X_train, y_train)
y_logreg = model.predict(X_test)
Logregpredict.append(y_logreg)

#KNN
knclassifier = KNeighborsClassifier(n_neighbors=40)
knclassifier.fit(X_train, y_train)
y_est = knclassifier.predict(X_test)
KNNpredict.append(y_est)

#baseline
Baselinepredict.append(np.ones(len(X_test)))

#print(mcnemar(y[1],Logregpredict[0],KNNpredict[0]))
print(mcnemar(y[1],Logregpredict[0],Baselinepredict))
print(mcnemar(y[1],Baselinepredict,KNNpredict[0]))