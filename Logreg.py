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
from __init__ import train_neural_net, draw_neural_net
from scipy import stats
from clean_data import clean_data, transform_data

#-----------------------LOADING DATA----------------------------

data = clean_data('Datasets/**videos.csv')
data = transform_data(data,['likes','dislikes','views','comment_count','trending_time'])
np.random.seed(180820)
data = data.head(100000)
X = np.array(data[['likes','dislikes','views','comment_count','trending_time']])
#y = np.array(data['views']).squeeze()
data['class'] = np.where(data["trending_time"]<=3., 1, 0.)
y = np.where(data["trending_time"]<=3., 1, 0.)
#X = np.array(data)
#y = X[:,[4]]             
#X = X[:,0:4]
attributeNames = ['likes','dislikes','views','comment_count','trending_time']
N, M = X.shape

print(data['class'].value_counts())
#----------------------TAKEN FROM EXERCISE 5.2.6------------------

# Fit logistic regression model
def logReg():
    
    K_o_splits = 10 #Ydre opdeling (K1)
    outer_it = 0 
    K_i_splits = 10 # Indre opdeling (K2)
    model_count=10

    summed_eval_i = np.zeros((model_count)) #bruges til at gemme summerne af Eval_M_s_j
    eval_i = np.zeros((model_count))
    eval_o = np.zeros((model_count))
    optimal_lambda = np.zeros((K_o_splits))

    CV1 = model_selection.KFold(n_splits=K_o_splits,shuffle=True)
    CV2 = model_selection.KFold(n_splits=K_i_splits,shuffle=True)
    
    errors_inner=[]
    errors_outer=[]
    
    for train_index_o, test_index_o in CV1.split(X,y):
        print('Outer CV1-fold {0} of {1}'.format(outer_it+1,K_o_splits))
        # extract training and test set for current CV fold
        X_train_o = X[train_index_o,:]
        y_train_o = y[train_index_o]
        X_test_o = X[test_index_o,:]
        y_test_o = y[test_index_o]

        inner_it = 0
        for train_index_i, test_index_i in CV2.split(X,y):
            print('Inner CV2-fold {0} of {1}'.format(inner_it+1,K_i_splits))
            X_train_i = X[train_index_i,:]
            y_train_i = y[train_index_i]
            X_test_i = X[test_index_i,:]
            y_test_i = y[test_index_i]

            for lambd in range(10):
                
                reg_term =  1+lambd*3 #Det er noget i selv lige skal finde et passende interval for
            
                #KNN
                #knclassifier = KNeighborsClassifier(n_neighbors=40);
                #knclassifier.fit(X_train_i, y_train_i);
                #y_est = knclassifier.predict(X_test_i);
                #current_err = 100*( (y_est!=y_test_i).sum().astype(float)/ len(y_test_i))
                
                #Logistic regression
                model = lm.logistic.LogisticRegression(C=reg_term,penalty='l2')
                model = model.fit(X_train_i, y_train_i)
                y_logreg = model.predict(X_test_i)
                current_err = 100*(y_logreg!=y_test_i).sum()/len(y_test_i)
                
                #DecisionTree                
                #model2 = tree.DecisionTreeClassifier(max_depth=reg_term,criterion = "entropy") ###NEED REGU
                #model2 = model2.fit(X_train_i, y_train_i)
                #y_dectree = model2.predict(X_test_i)
                #current_err = 100*(y_dectree!=y_test_i).sum().astype(float)/len(y_test_i)
                errors_inner.append(current_err)
                summed_eval_i[lambd] += current_err
            
            inner_it += 1

        eval_i = summed_eval_i * (len(X_test_i)/len(X_train_o))     
        idx = np.argmin(eval_i) #Finde ud af hvilken model s, var den bedste
        reg_term = (1+idx*3) #Genskaber den reguleringsværdi (Eksempelvis hvilken lambda værdi, der blev brugt ved den bedste model)
        
        model = lm.logistic.LogisticRegression(C=reg_term,penalty='l2')
        model = model.fit(X_train_o, y_train_o)
        y_logreg = model.predict(X_test_o)
        current_err = 100*(y_logreg!=y_test_o).sum()/len(y_test_o) #Hvor god er den bedste model s til at forudsige det yderste test split (D_i_test)
        errors_outer.append(current_err)
        
        eval_o[outer_it] = current_err #Gem E_s_gen
        optimal_lambda[outer_it] = reg_term
        
        
        outer_it+=1
        
    mode_reg, _= np.unique(optimal_lambda, return_counts=True) 
    

    figure()
    boxplot(eval_o)
    xlabel('Logistic Reggression')
    ylabel('Cross-validation error [%]')
    show()
    e_gen = np.sum(eval_o) * (len(X_test_o)/ len(X))
    print("Logreg generalization error: %f with %s and %i" % (e_gen,'lambda',mode_reg[0]))

    #print(errors_inner)
    print(errors_outer)
    print(reg_term)
    print(mode_reg)
    print(optimal_lambda)


print(logReg())


