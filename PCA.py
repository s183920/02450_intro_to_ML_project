from clean_data import clean_data
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.linalg import svd
from categories import find_cat_labels

df = clean_data('Datasets/**videos.csv')

#remove outliers
def remove_outlier(df, series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3-q1
    lower_bound = q1-1.5*iqr
    upper_bound = q3 + 1.5*iqr
    df = df[series >= lower_bound]
    df = df[series <= upper_bound]
    
    return df
"""
for colname in df.columns:
     if np.issubdtype(df[colname].dtype, np.number):
        df = remove_outlier(df, df[colname]) #removes outliers from every column
"""
#df = remove_outlier(df, df.trending_time) #only removes outliers from trending time, as this seems to be the problem when looking at a pairplot (see summary.py)
#df = df.reset_index()


# normalize data
data_to_be_norm = df[["likes", "dislikes", "views", "comment_count"]]#, "trending_time"]]
y = np.log(data_to_be_norm)
y[y == -np.inf] = 0
y[y == np.inf] = 0
data_normalized =  (y- data_to_be_norm.mean(axis = 0)) /  data_to_be_norm.std(axis = 0)

# PCA!!!
U,S,Vh = svd(data_normalized,full_matrices=False)
V = Vh.T  

print(pd.DataFrame(V).to_latex())
proj_data = data_normalized @ V


# plot

index = [df.category_id[id] in [10, 15,20,22,26,30] for id in range(len(df.category_id))]

proj_plot_data = proj_data[index]


i, j = 0, 1 # principal components



scatter = plt.scatter(proj_plot_data[i], proj_plot_data[j], c = df.category_id[index], alpha = 0.5, cmap = "viridis", s = 2)
cat_labels = find_cat_labels('Datasets/CA_category_id.json')
classLabels = [cat_labels[i] for i in ["10","15","20","22","26","30"]] 
labels = np.unique(df.category_id[index])
handles = [plt.Line2D([],[],marker="o", ls="", 
                      color=scatter.cmap(scatter.norm(yi))) for yi in labels]
plt.legend(handles, classLabels)
plt.xlabel("Principal component {}".format(i+1))
plt.ylabel("Principal component {}".format(j+1))
plt.title("Visualisation of our data with log-transformation before PCA")


plt.show()

# explained variance
vars = proj_plot_data.apply(np.var, axis = 0)
explained_vars = vars/sum(vars)

print(explained_vars.to_latex())
