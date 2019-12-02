from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from scipy.io import loadmat
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
from scipy.io import loadmat
import torch
from sklearn import model_selection
from __init__ import train_neural_net, draw_neural_net
from scipy import stats
from clean_data import clean_data, transform_data

# SETUP ------------------------------------------------------------------------------------------------------
cols = ["likes", "dislikes", "views", "comment_count", "trending_time"]
data = clean_data('Datasets/**videos.csv')
data = data.head(5000)
data_norm = transform_data(data, cols)

data["class"] = np.where(data["trending_time"]<=3., 1, 0.)
#Data
X = np.array(data_norm[["likes", "dislikes", "views", "comment_count", "trending_time"]])
y = np.array(data["class"]).squeeze()
attributeNames = ["likes", "dislikes", "views", "comment_count", "trending_time"]
N, M = X.shape

#Bruger K-fold crossvalidation til at estimatere antallet af naboer k, til k-nearest neighbour classifier

#k-fold cross validation with classifier 
# Create crossvalidation partition for evaluation

K1_split = 10
K2_split = 10


L = 40 #n neighbours
CV1 = model_selection.KFold(n_splits=K1_split, shuffle=True)
CV2 = model_selection.KFold(n_splits=K2_split,shuffle=True)
summed_eval_i = np.zeros((model_count))


cur_err = np.zeros((N,L))
k1=0
for train_index, test_index in CV1.split(X,y):
    print('Computing CV fold: {0}/{1}..'.format(k1+1,K1_split))
#extract training and test set for current CV fold
    X_train, y_train = X[train_index,:], y[train_index]
    X_test, y_test = X[test_index,:], y[test_index]
	
	k2 = 0
	
	for train_index_i, test_index_i in CV2.split(X_train,y_train):
	   print('Inner CV2-fold {0} of {1}'.format(k2+1,K2_split))
	   X_train_i = X[train_index_i,:], y_train_i = y[train_index_i]
	   X_test_i = X[test_index_i,:], y_test_i = y[test_index_i]
	   
	   for l in range(1,L+1):
		   knclassifier = KNeighborsClassifier(n_neighbors=l)
		   knclassifier.fit(X_train_i, y_train_i)
		   y_est = knclassifier.predict(X_test_i)
		   cur_err[k2,l-1] = np.sum(y_est!=y_test_i)
		   
		   k2+=1
		   
		   min_err = np.min(cur_err)
		   
		   
		   summed_eval_i[k2] += min_err
	
	
	
	#low_err_idx = np.argmin(curr_err) #Neighbour val. 
	e_gen = np.sum(eval_o) * (len(X_test)/ len(X))
	
	#The average of the e_gens. 