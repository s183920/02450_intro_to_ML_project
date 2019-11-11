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

# Load Matlab data file and extract variables of interest
data = clean_data('Datasets/**videos.csv')
data = transform_data(data,['likes','dislikes','views','comment_count','trending_time'])
data = data.head(6000) #viser at man sagtens kan plotte training error med mindre data
#X = np.array(data[['likes','dislikes','comment_count','trending_time']])
#y = np.array(data['views']).squeeze()
X = np.array(data)
y = X[:,[4]]             # alcohol contents (target)
X = X[:,0:4]
attributeNames = ['likes','dislikes','views','comment_count','trending_time',]
N, M = X.shape
C = 2


# Parameters for neural network classifier
n_hidden_units = 5      # number of hidden units
n_replicates = 1      # number of networks trained in each k-fold
max_iter = 10000        # 

# K-fold crossvalidation
K = 10                  # only three folds to speed up this example
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

errors = np.asarray(errors)
#errors.resize((4,4)).shape()
Fail = {'Hidden Units':[1,2,3,4,5],
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
df = DataFrame(Fail,columns=['Hidden Units','Cv-Fold1','Cv-Fold2','Cv-Fold3',
                             'Cv-Fold4','Cv-Fold5','Cv-Fold6','Cv-Fold7',
                             'Cv-Fold8','Cv-Fold9','Cv-Fold10'])
print(df)
"""
    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')
# Display the MSE across folds
    
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors)), color=color_list)
summaries_axes[1].set_xlabel('Fold');
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE');
summaries_axes[1].set_title('Test mean-squared-error')
    
print('Diagram of best neural net in last fold:')
weights = [net[i].weight.data.numpy().T for i in [0,2]]
biases = [net[i].bias.data.numpy() for i in [0,2]]
tf =  [str(net[i]) for i in [1,2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames)

# Print the average classification error rate
print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value

plt.figure(figsize=(10,10));
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy();
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Trending time prediction: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()

"""
