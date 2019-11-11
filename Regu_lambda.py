# exercise 8.1.1

from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from __init__ import rlr_validate
from clean_data import clean_data, transform_data
import pandas as pd
import matplotlib.pyplot as plt

# SETUP ------------------------------------------------------------------------------------------------------
cols = ["likes", "dislikes", "views", "comment_count", "trending_time"]
data = clean_data('Datasets/**videos.csv')
data_norm = transform_data(data, cols)

np.random.seed(180820)

"""
index = np.random.choice(range(0, len(data_norm)), size = 10000, replace = False)
index = data_norm.index in index
data_norm = data_norm[index,:]
"""


#data_norm.head(100) #viser at man sagtens kan plotte training error med mindre data

X = np.array(data_norm[["likes", "dislikes","views", "comment_count"]])#, "trending_time"]])
y = np.array(data_norm["trending_time"]).squeeze()
attributeNames = ["likes", "dislikes", "views", "comment_count"]#, "trending_time"]
N, M = X.shape

# options------------------------------------------------------------------------------------------------------
K1 = 10 #outerfold
K2 = 10 #innerfold
layer2 = True #whether to do 2 layer cross validation or not

start_lambda = 1
end_lambda = 22
step_size = 1

print_info = True #whether or not to print info for outer loop
reg_plots = True # whether or not to plot regularisation plots

# Add offset attribute ------------------------------------------------------------------------------------------------------
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
attributeNames = [u'Offset']+attributeNames
M = M+1

## Crossvalidation outer ------------------------------------------------------------------------------------------------------
# Create crossvalidation partition for evaluation
K = K1  
CV = model_selection.KFold(K, shuffle=True)

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
    for train_index, test_index in CV.split(X,y):
        
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
            figure(1, figsize=(12,8))
            subplot(1,2,1)
            plt.plot(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
            xlabel('Regularization factor')
            ylabel('Mean Coefficient Values')
            grid()
            # You can choose to display the legend, but it's omitted for a cleaner 
            # plot, since there are many attributes
            #legend(attributeNames[1:], loc='best')
            
            subplot(1,2,2)
            title('Optimal lambda: {0}'.format(opt_lambda))
            #loglog(lambdas,train_err_vs_lambda.T,'b.-', alpha=0.5, linewidth=10)
            #loglog(lambdas,test_err_vs_lambda.T,'r.-')
            plt.plot(lambdas, train_err_vs_lambda.T, "b.-")
            plt.plot(lambdas, test_err_vs_lambda.T, "r.-")
            xlabel('Regularization factor')
            ylabel('Squared error (crossvalidation)')
            legend(['Train error','Validation error'])
            grid()
            
            show()

            # train and test error individually 
            figure(2, figsize=(12,8))
            title('Optimal lambda: {0}'.format(opt_lambda))
            subplot(1,2,1)
            title("Train data")
            #loglog(lambdas,train_err_vs_lambda.T,'b.-', alpha=0.5, linewidth=10)
            #loglog(lambdas,test_err_vs_lambda.T,'r.-')
            plt.plot(lambdas, train_err_vs_lambda.T, "b.-")
            #plt.plot(lambdas, test_err_vs_lambda.T, "r.-")
            xlabel('Regularization factor')
            ylabel('Squared error (crossvalidation)')
            #legend(['Train error','Validation error'])
            grid()

            subplot(1,2,2)
            title("Test data")
            #loglog(lambdas,train_err_vs_lambda.T,'b.-', alpha=0.5, linewidth=10)
            #loglog(lambdas,test_err_vs_lambda.T,'r.-')
            #plt.plot(lambdas, train_err_vs_lambda.T, "b.-")
            plt.plot(lambdas, test_err_vs_lambda.T, "r.-")
            xlabel('Regularization factor')
            ylabel('Squared error (crossvalidation)')
            #legend(['Train error','Validation error'])
            grid()

            show()

            # weights individually
            figure(3, figsize=(12,16))
            subplot(2,2,1)
            title(attributeNames[1])
            plt.plot(lambdas,mean_w_vs_lambda.T[:,1],'.-') # Don't plot the bias term
            xlabel('Regularization factor')
            ylabel('Mean Coefficient Values')
            grid()

            subplot(2,2,2)
        
            title(attributeNames[2])
            plt.plot(lambdas,mean_w_vs_lambda.T[:,2],'.-') # Don't plot the bias term
            xlabel('Regularization factor')
            ylabel('Mean Coefficient Values')
            grid()

            subplot(2,2,3)
            title(attributeNames[3])
            plt.plot(lambdas,mean_w_vs_lambda.T[:,3],'.-') # Don't plot the bias term
            xlabel('Regularization factor')
            ylabel('Mean Coefficient Values')
            grid()

            subplot(2,2,4)
            title(attributeNames[4])
            plt.plot(lambdas,mean_w_vs_lambda.T[:,4],'.-') # Don't plot the bias term
            xlabel('Regularization factor')
            ylabel('Mean Coefficient Values')
            grid()

            plt.subplots_adjust(wspace = 0.4)
            show()
        
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
    print(error_df.to_latex())


else:
    k = 0
    # extract training and test set for current CV fold
    X_train = X
    y_train = y
    X_test = X
    y_test = y

    
    internal_cross_validation = K2    
    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train, lambdas, internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    
    mu = np.mean(X_train[:, 1:], 0)
    sigma = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu ) / sigma 
    X_test[:, 1:] = (X_test[:, 1:] - mu ) / sigma
    
    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train
    
    # Compute mean squared error without using the input data at all
    Error_train_nofeatures = np.square(y_train-y_train.mean()).sum(axis=0)/y_train.shape[0]
    Error_test_nofeatures = np.square(y_test-y_test.mean()).sum(axis=0)/y_test.shape[0]


    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0,0] = 0 # Do no regularize the bias term
    w_rlr = np.linalg.solve(XtX+lambdaI,Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr = np.square(y_train-X_train @ w_rlr).sum(axis=0)/y_train.shape[0]
    Error_test_rlr = np.square(y_test-X_test @ w_rlr).sum(axis=0)/y_test.shape[0]

    # Estimate weights for unregularized linear regression, on entire training set
    w_noreg = np.linalg.solve(XtX,Xty).squeeze()
    # Compute mean squared error without regularization
    Error_train = np.square(y_train-X_train @ w_noreg).sum(axis=0)/y_train.shape[0]
    Error_test = np.square(y_test-X_test @ w_noreg).sum(axis=0)/y_test.shape[0]

    # Display the results for the last cross-validation fold
    if reg_plots:
        figure(1, figsize=(12,8))
        subplot(1,2,1)
        plt.plot(lambdas,mean_w_vs_lambda.T[:,1:],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()
        # You can choose to display the legend, but it's omitted for a cleaner 
        # plot, since there are many attributes
        #legend(attributeNames[1:], loc='best')
        
        subplot(1,2,2)
        title('Optimal lambda: {0}'.format(opt_lambda))
        #loglog(lambdas,train_err_vs_lambda.T,'b.-', alpha=0.5, linewidth=10)
        #loglog(lambdas,test_err_vs_lambda.T,'r.-')
        plt.plot(lambdas, train_err_vs_lambda.T, "b.-")
        plt.plot(lambdas, test_err_vs_lambda.T, "r.-")
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        legend(['Train error','Validation error'])
        grid()
        
        show()

        # train and test error individually 
        figure(2, figsize=(12,8))
        title('Optimal lambda: {0}'.format(opt_lambda))
        subplot(1,2,1)
        title("Train data")
        #loglog(lambdas,train_err_vs_lambda.T,'b.-', alpha=0.5, linewidth=10)
        #loglog(lambdas,test_err_vs_lambda.T,'r.-')
        plt.plot(lambdas, train_err_vs_lambda.T, "b.-")
        #plt.plot(lambdas, test_err_vs_lambda.T, "r.-")
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        #legend(['Train error','Validation error'])
        grid()

        subplot(1,2,2)
        title("Test data")
        #loglog(lambdas,train_err_vs_lambda.T,'b.-', alpha=0.5, linewidth=10)
        #loglog(lambdas,test_err_vs_lambda.T,'r.-')
        #plt.plot(lambdas, train_err_vs_lambda.T, "b.-")
        plt.plot(lambdas, test_err_vs_lambda.T, "r.-")
        xlabel('Regularization factor')
        ylabel('Squared error (crossvalidation)')
        #legend(['Train error','Validation error'])
        grid()

        show()

        # weights individually
        figure(3, figsize=(12,16))
        subplot(2,2,1)
        title(attributeNames[1])
        plt.plot(lambdas,mean_w_vs_lambda.T[:,1],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()

        subplot(2,2,2)
    
        title(attributeNames[2])
        plt.plot(lambdas,mean_w_vs_lambda.T[:,2],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()

        subplot(2,2,3)
        title(attributeNames[3])
        plt.plot(lambdas,mean_w_vs_lambda.T[:,3],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()

        subplot(2,2,4)
        title(attributeNames[4])
        plt.plot(lambdas,mean_w_vs_lambda.T[:,4],'.-') # Don't plot the bias term
        xlabel('Regularization factor')
        ylabel('Mean Coefficient Values')
        grid()

        plt.subplots_adjust(wspace = 0.4, hspace = 0.4)
        show()
    
    print('Weights in optimal fold:')
    opt_index = np.where(lambdas == opt_lambda)
    for m in range(M):
        print('{}: {}'.format(attributeNames[m], mean_w_vs_lambda.T[opt_index,m]))
    print('Linear regression without feature selection:')








if layer2 == True:
    print('Weights in last fold:')
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m], w_rlr[m,-1]))
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
else:
    print('Weights in last fold:')
    for m in range(M):
        print('{:>15} {:>15}'.format(attributeNames[m], w_rlr[m]))
    # Display results
    print("Opt error: {}". format(opt_val_err))

    



#print('Ran Exercise 8.1.1')