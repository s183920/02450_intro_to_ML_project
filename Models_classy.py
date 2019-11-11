import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from clean_data import clean_data, transform_data

import sklearn.linear_model as lm
from sklearn import model_selection
from __init__ import feature_selector_lr, bmplot
from scipy import stats
from clean_data import clean_data, transform_data

#SETUO
cols = ["likes", "dislikes", "views", "comment_count", "trending_time"]
data = clean_data('Datasets/**videos.csv')
data_norm = transform_data(data, cols)

data["class"] = np.where(data["trending_time"]<=3., 1, 0.)
#Data
X = np.array(data_norm[["likes", "dislikes", "views", "comment_count", "trending_time"]])
y = np.array(data["class"]).squeeze()
attributeNames = ["likes", "dislikes", "views", "comment_count", "trending_time"]
N, M = X.shape



