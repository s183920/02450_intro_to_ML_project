# From exercise 6.3.2

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
data = data.head(100000)
data_norm = transform_data(data, cols)

data["class"] = np.where(data["trending_time"]<=3., 1, 0.)
#Data
X = np.array(data_norm[["likes", "dislikes", "views", "comment_count", "trending_time"]])
y = np.array(data["class"]).squeeze()
attributeNames = ["likes", "dislikes", "views", "comment_count", "trending_time"]
N, M = X.shape


#Bruger K-fold crossvalidation til at estimatere antallet af naboer k, til k-nearest neighbour classifier

#k-fold cross validation with classifier 

K = 10

CV = model_selection.KFold(K, shuffle=True)
L=40 # Maximum number of neighbors


errors = np.zeros((N,L))
K = 10
internal_cross_validation = 10

k1=0
for train_index, test_index in CV1.split(X,y):
    
    # extract training and test set for current CV fold
    Xo_train = X[train_index,:]
    yo_train = y[train_index]
    Xo_test = X[test_index,:]
    yo_test = y[test_index]
	CV2 = model_selection.KFold(internal_cross_validation, shuffle=True)
	
	k2 = 0
    for train_index, test_index in CV2.split(Xo_train,yo_train):
        Xi_train = Xo_train[train_index,:]
        yi_train = yo_train[train_index]
        Xi_test = Xo_train[test_index,:]
        yi_test = yo_train[test_index]
		
		k3=0
        for l in range(1,L+1):
            knclassifier = KNeighborsClassifier(n_neighbors=l).fit(Xi_train, yi_train)
            knn_pred = np.array(knclassifier.predict(Xi_test))
			errors[k,l-1] = np.sum(knn_pred!=yi_test)/N
			
		k3+=1

		print('Inner '+str(k2))
		k2+=1
        
	print('Outer ' +str(k1))    
	k1+=1

summary = np.empty((K,5))



#errors = np.zeros((N,L))
#k=0
#for train_index, test_index in CV.split(X):
#    print('Computing CV fold: {0}/{1}..'.format(k+1,K))

    # extract training and test set for current CV fold
 #   X_train, y_train = X[train_index,:], y[train_index]
 #   X_test, y_test = X[test_index,:], y[test_index]

	# Fit classifier and classify the test points (consider 1 to 40 neighbors)
    for l in range(1,L+1):
        knclassifier = KNeighborsClassifier(n_neighbors=l)
        knclassifier.fit(X_train, y_train)
        y_est = knclassifier.predict(X_test)
        errors[k,l-1] = np.sum(y_est!=y_test)

    k+=1

e_gen = np.sum(errors) * (len(X_test)/ len(X))

#sorted list of index
res = errors
sort_index = np.argsort(res)

errors = 100*sum(errors,0)/N
print("Error percent ", errors)
print("Gen_Err")
print(np.sum(errors))

figure()
plot(errors)
xlabel('Number of neighbors')
ylabel('Classification error rate (%)')
show()



