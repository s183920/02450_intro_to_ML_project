# exercise 8.2.6
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import torch
from sklearn import model_selection
from __init__ import train_neural_net, draw_neural_net
from scipy import stats
from clean_data import *
from pandas import DataFrame
import pandas as pd

# Load Matlab data file and extract variables of interest
data = clean_data('Datasets/**videos.csv')
data = transform_data(data,['likes','dislikes','views','comment_count','trending_time'])
np.random.seed(180820)
data = data.head(10000)
#X = np.array(data[['likes','dislikes','comment_count','trending_time']])
#y = np.array(data['views']).squeeze()
X = np.array(data)
y = X[:,[4]]             
X = X[:,0:4]
attributeNames = ['likes','dislikes','views','comment_count','trending_time']
N, M = X.shape
C = 2


# Parameters for neural network classifier
n_hidden_units = 5      # number of hidden units
n_replicates = 1      # number of networks trained in each k-fold
max_iter = 10000        # 

# K-fold crossvalidation
K = 10                 # only three folds to speed up this example
CV = model_selection.KFold(K, shuffle=True)

# Setup figure for display of learning curves and error rates in fold
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']
# Define the model


#print('Training model of type:\n\n{}\n'.format(str(model())))
errors = [] # make a list for storing generalizaition error in each loop
for (k, (train_index, test_index)) in enumerate(CV.split(X,y)): 
    print('\nCrossvalidation fold: {0}/{1}'.format(k+1,K))    
    
    # Extract training and test set for current CV fold, convert to tensors
    X_train = torch.tensor(X[train_index,:], dtype=torch.float)
    y_train = torch.tensor(y[train_index], dtype=torch.float)
    X_test = torch.tensor(X[test_index,:], dtype=torch.float)
    y_test = torch.tensor(y[test_index], dtype=torch.uint8)
    #Der skal være et for loop der tager den optimale model/ antal hidden units. 
    for h in range(1,11):
        print('Hidden Layer nr. {}'.format(h))
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

#Reshaping our errors to work with a Dataframe
errors = np.asarray(errors)
errors = errors.reshape(10,10)

#print(errors[:,0])
#print(errors[:,1])
#print(errors[:,2])

#Dataframe
Fails = {'Cv-Folds':[1,2,3,4,5,6,7,8,9,10],
         'H1': errors[:,0],
         'H2': errors[:,1],
         'H3': errors[:,2],
         'H4': errors[:,3],
         'H5': errors[:,4],
         'H6': errors[:,5],
         'H7': errors[:,6],
         'H8': errors[:,7],
         'H9': errors[:,8],
         'H10':errors[:,9]
         }
Errordf = DataFrame(Fails,columns=['Cv-Folds','H1','H2','H3','H4','H5','H6','H7','H8','H9','H10'])

with pd.option_context('display.max_rows', None, 'display.max_columns', None): 
    print(Errordf)

#finding the minimum error, and the number of hidden units for that error. 
minValues = Errordf.min()
minIndex = Errordf.idxmin(axis=0)

WubWub = {'hidden units': minIndex+1,
          'E-test': minValues}
endData = DataFrame(WubWub, columns=['hidden units','E-test'])
print(endData)

