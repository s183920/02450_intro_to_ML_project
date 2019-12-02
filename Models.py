import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from __init__ import train_neural_net, draw_neural_net, rlr_validate
from scipy import stats
from clean_data import *
from pandas import DataFrame
import pandas as pd
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import sklearn.linear_model as lmq

# -----------------------SETUP--------------------------------------------------

# Load Data from our script
data = clean_data('Datasets/**videos.csv')
data = transform_data(data,['likes','dislikes','views','comment_count','trending_time'])
<<<<<<< HEAD
data = data.head(200) #viser at man sagtens kan plotte training error med mindre data
X = np.array(data[['likes','dislikes','comment_count','trending_time']])
y = np.array(data['views']).squeeze()
=======
data = data.head(100) #viser at man sagtens kan plotte training error med mindre data
#X = np.array(data[['likes','dislikes','comment_count','trending_time']])
#y = np.array(data['views']).squeeze()
X = np.array(data)
y = X[:,[4]]             # Trending time prediction
X = X[:,0:4]
>>>>>>> c311ba3134be21e7919ba6e95f686812451253d2
attributeNames = ['likes','dislikes','views','comment_count','trending_time',]
N, M = X.shape
C = 2

<<<<<<< HEAD
=======

>>>>>>> c311ba3134be21e7919ba6e95f686812451253d2

#-------------------LINEAR REGRESSION and BASELINE-----------------------------
# options
K1 = 10 #outerfold
K2 = 10 #innerfold
layer2 = True #whether to do 2 layer cross validation or not

start_lambda = -1
end_lambda = 50
step_size = 1

print_info = True #whether or not to print info for outer loop
reg_plots = False # whether or not to plot regularisation plots

# Add offset attribute ------------------------------------------------------------------------------------------------------
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation outer ------------------------------------------------------------------------------------------------------
# Create crossvalidation partition for evaluation
K = K1  
CV = model_selection.KFold(K, shuffle=True)
splits = CV.split(X,y)

# Values of lambda ------------------------------------------------------------------------------------------------------
#lambdas = np.power(10.,range(-2,10))
lambdas = np.linspace(start_lambda, end_lambda, (end_lambda-start_lambda)/step_size +1 )


# Initialize variables ------------------------------------------------------------------------------------------------------
#T = len(lambdas)
Error_train = np.empty((K,1))
Error_test = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
w_rlr = np.empty((M,K))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
w_noreg = np.empty((M,K))

# Outer loop ------------------------------------------------------------------------------------------------------
if layer2:
    k=0
    for train_index, test_index in splits:
        
        # extract training and test set for current CV fold
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]

        
        internal_cross_validation = K2    
        opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

        # Standardize outer fold based on training set, and save the mean and standard
        # deviations since they're part of the model (they would be needed for
        # making new predictions) - for brevity we won't always store these in the scripts
        mu[k, :] = np.mean(X_train[:, 1:], 0)
        sigma[k, :] = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
        X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :] 
        
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        
        # Compute mean squared error without using the input data at all
        Error_train_nofeatures[k] = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
        Error_test_nofeatures[k] = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]

        baseline = np.square(y_test-y_train.mean()).sum(axis=0)/y_test.shape[0]

        # Estimate weights for the optimal value of lambda, on entire training set
        lambdaI = opt_lambda * np.eye(M)
        lambdaI[0,0] = 0 # Do no regularize the bias term
        w_rlr[:,k] = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
        # Compute mean squared error with regularization with optimal lambda
        Error_train_rlr[k] = np.square(y_train-X_train @ w_rlr[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test_rlr[k] = np.square(y_test-X_test @ w_rlr[:,k]).sum(axis=0)/y_test.shape[0]

        # Estimate weights for unregularized linear regression, on entire training set
        w_noreg[:,k] = np.linalg.solve(XtX,Xty).squeeze()
        # Compute mean squared error without regularization
        Error_train[k] = np.square(y_train-X_train @ w_noreg[:,k]).sum(axis=0)/y_train.shape[0]
        Error_test[k] = np.square(y_test-X_test @ w_noreg[:,k]).sum(axis=0)/y_test.shape[0]

        # Display the results for the last cross-validation fold
        if k == K-1 and reg_plots:
            figure(k, figsize=(12,8))
            subplot(1,2,1)
            semilogx(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
            xlabel('Regularization factor')
            ylabel('Mean Coefficient Values')
            grid()
            # You can choose to display the legend, but it's omitted for a cleaner 
            # plot, since there are many attributes
            #legend(attributeNames[1:], loc='best')
            
            subplot(1,2,2)
            title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
            loglog(lambdas,train_err_vs_lambda.T,'b.-', alpha=0.5, linewidth=10)
            loglog(lambdas,test_err_vs_lambda.T,'r.-')
            xlabel('Regularization factor')
            ylabel('Squared error (crossvalidation)')
            legend(['Train error','Validation error'])
            grid()
        
        # To inspect the used indices, use these print statements
        #print('Cross validation fold {0}/{1}:'.format(k+1,K))
        #print('Train indices: {0}'.format(train_index))
        #print('Test indices: {0}\n'.format(test_index))

        #print info ------------------------------------------------------------------------------------------------------
        
        if print_info:
            print('Cross validation fold {0}/{1}:'.format(k+1,K))
            print("Lambda = {}".format(opt_lambda))
            print("Generalisation error for linear regression= {}".format(test_err_vs_lambda.mean()))
            print("Baseline error = {}".format(baseline))

        # Store errors
        error_names = ["Outer_fold", "lambda_i", "Lin_reg_error", "Baseline_error"]
        error_data = np.array([k+1, opt_lambda, test_err_vs_lambda.mean(), baseline])

        if k == 0:
            error_array = error_data
        else:
            error_array = np.vstack((error_array, error_data))

        k+=1

    error_df = pd.DataFrame(error_array, columns = error_names)
    print(error_df)
show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test.sum())/Error_test_nofeatures.sum()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

print('Weights in last fold:')
for m in range(M):
    print('{:>15} {:>15}'.format(attributeNames[m], np.round(w_rlr[m,-1],2)))


#--------------------------ANN------------------------------------
X = np.array(data)
y = X[:,[4]]             # Trending time prediction
X = X[:,0:4]
splits = CV.split(X,y)
# Parameters for neural network classifier
n_replicates = 1      # number of networks trained in each k-fold
max_iter = 10000        # 

# K-fold crossvalidation
K = K1                  # only three folds to speed up this example
#CV = model_selection.KFold(K, shuffle=True)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
# Define the model


#print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
for (k, (train_index, test_index)) in enumerate(splits): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train = torch.tensor(y[train_index], dtype=torch.float)
    X_test = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test = torch.tensor(y[test_index], dtype=torch.uint8)
    #Der skal være et for loop der tager den optimale model/ antal hidden units. 
    for h in range(1,11):
        print('Hidden unit nr. {}'.format(h))
        n_hidden_units = h
        model = lambda: torch.nn.Sequential(
                    torch.nn.Linear(M, n_hidden_units), #M features to n_hidden_units
                    torch.nn.Tanh(),   # 1st transfer function,
                    torch.nn.Linear(n_hidden_units, 1), # n_hidden_units to 1 output neuron
                    # no final tranfer function, i.e. "linear output"
                    )
        loss_fn = torch.nn.MSELoss() # notice how this is now a mean-squared-error loss

        # Train the net on training data
        net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)
    
        print('\n\tBest loss: {}\n'.format(final_loss))



        # Determine estimated class labels for test set
        y_test_est = net(X_test)
    
        # Determine errors and errors
        se = (y_test_est.float()-y_test.float())**2 # squared error
        mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
        errors.append(mse) # store error rate for current CV fold 
    #Generr.append(round(np.sqrt(np.mean(errors))))

errors = np.asarray(errors)
print(errors)
#errors.resize((4,4)).shape()
'''
Fail = {'Hidden Units':[1,2,3,4,5,6,7,8,9,10],
        'Cv-Fold1':[errors[0],errors[10],errors[20],errors[30],errors[40],errors[50],errors[60],errors[70],errors[80],errors[90]],
        'Cv-Fold2':[errors[1],errors[11],errors[21],errors[31],errors[41],errors[51],errors[61],errors[71],errors[81],errors[91]],
        'Cv-Fold3':[errors[2],errors[12],errors[22],errors[32],errors[42],errors[52],errors[62],errors[72],errors[82],errors[92]],
        'Cv-Fold4':[errors[3],errors[13],errors[23],errors[33],errors[43],errors[53],errors[63],errors[73],errors[83],errors[93]],
        'Cv-Fold5':[errors[4],errors[14],errors[24],errors[34],errors[44],errors[54],errors[64],errors[74],errors[84],errors[94]],
        'Cv-Fold6':[errors[5],errors[15],errors[25],errors[35],errors[45],errors[55],errors[65],errors[75],errors[85],errors[95]],
        'Cv-Fold7':[errors[6],errors[16],errors[26],errors[36],errors[46],errors[56],errors[66],errors[76],errors[86],errors[96]],
        'Cv-Fold8':[errors[7],errors[17],errors[27],errors[37],errors[47],errors[57],errors[67],errors[77],errors[87],errors[97]],
        'Cv-Fold9':[errors[8],errors[18],errors[28],errors[38],errors[48],errors[58],errors[68],errors[78],errors[88],errors[98]],
        'Cv-Fold10':[errors[9],errors[19],errors[29],errors[39],errors[49],errors[59],errors[69],errors[79],errors[89],errors[99]]
        }
Errordf = DataFrame(Fail,columns=['Hidden Units','Cv-Fold1','Cv-Fold2','Cv-Fold3','Cv-Fold4','Cv-Fold5','Cv-Fold6','Cv-Fold7','Cv-Fold8','Cv-Fold9','Cv-Fold10'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(Errordf)


Fails = {'Cv-Folds':[1,2,3,4,5,6,7,8,9,10],
         'H1':[errors[0:10]],
         'H2':[errors[10:20]],
         'H3':[errors[20:30]],
         'H4':[errors[30:40]],
         'H5':[errors[40:50]],
         'H6':[errors[50:60]],
         'H7':[errors[60:70]],
         'H8':[errors[70:80]],
         'H9':[errors[80:90]],
         'H10':[errors[90:100]]
         }
Errordf = DataFrame(Fails,columns=['Cv-Folds','H1','H2','H3','H4','H5','H6','H7','H8','H9','H10'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(Errordf)


minValues = Errordf.min()
minIndex = Errordf.idxmin(axis=0)

WubWub = {'hidden units': minIndex,
          'E-test': minValues}
endData = DataFrame(WubWub, columns=['hidden units','E-test'])
print(endData)

# ---------------------------- dataframe -----------------------------------------

<<<<<<< HEAD
error_data = pd.concat([endData, error_df],axis=1)
=======
error_data = pd.concat([endData, error_df], axis = 1)
>>>>>>> c311ba3134be21e7919ba6e95f686812451253d2

print(error_data)
print(error_data.to_latex())
'''