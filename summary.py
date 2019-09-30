from clean_data import clean_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns
#from PCA import remove_outlier

df = clean_data('Datasets/**videos.csv')
pd.set_option('display.float_format', lambda x: '%.3f' % x)


data_to_be_norm = df[["likes", "dislikes", "views", "comment_count"]]#, "trending_time"]]
y = np.log(data_to_be_norm)
y[y == -np.inf] = 0
y[y == np.inf] = 0
data_normalized =  (y- data_to_be_norm.mean(axis = 0)) /  data_to_be_norm.std(axis = 0)
df = data_normalized

#investigating with pairplot
def pairplot():
    sns.pairplot(df[["likes", "dislikes", "views", "comment_count"]])
    plt.savefig("pairplot.png")
    plt.show()

#summary statistics
def Sumstat():
    info =  df.info()
    stats = df.describe()
    print(stats.to_latex())

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(info, stats)

#df = remove_outlier(df, df.trending_time) #only removes outliers from trending time, as this seems to be the problem when looking at a pairplot (see summary.py)
#df = df.reset_index()

#Visualisation of statistics
#data_to_be_norm = df[["likes", "dislikes", "views", "comment_count", "trending_time"]]
#data_normalized =  (data_to_be_norm - data_to_be_norm.mean(axis = 0)) /  data_to_be_norm.std(axis = 0)
#only values
#df = data_normalized

##BOXPLOT

#Extract data to matrix X
#datVal = df.to_numpy()
def boxplt():
    dataVal = df.values
    plt.boxplot(dataVal)
    plt.title('Boxplot of attributes')
    plt.legend()
    plt.show()
#print(boxplt())
#boxplot(info) 
#show()

#boxplot(X)
#xticks(range(1,5),attributeNames)
#ylabel('cm')


##ATTRIBUTES NORMAL DISTRUTION
def normal():
    dataVal = df.values
    plt.figure(figsize=(8,7))
    attributeNames = list(df.columns)
    N = len(dataVal)
    M = len(attributeNames)
    for i in range(M):
        plt.subplot(2,2,i+1)
        plt.hist(dataVal[:,i], color=('blue'), bins=25)
        plt.xlabel(attributeNames[i])
        #plt.xlim(-1,6)
        #plt.ylim(0,10000)
    plt.subplots_adjust(hspace = 0.5)
    plt.show()

print(normal())

##CORRELATION BETWEEN ATTRIBUTES IN COVARIANCE MATRIX
#covMat = df.corr()
#print('CORRELATIONS', covMat)
#print(covMat.to_latex())





#print(pairplot())




