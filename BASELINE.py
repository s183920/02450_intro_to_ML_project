import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from clean_data import clean_data, transform_data
import matplotlib.pyplot as plt


#Data
#X = np.array(data_norm[["likes", "dislikes", "comment_count", "trending_time"]])
#y = np.array(data_norm["views"]).squeeze()
#attributeNames = ["likes", "dislikes", "comment_count", "trending_time"]
#N, M = X.shape

# SETUP ------------------------------------------------------------------------------------------------------
cols = ["likes", "dislikes", "views", "comment_count", "trending_time"]
data = clean_data('Datasets/**videos.csv')
data = data.head(100000)
data_norm = transform_data(data, cols)

data["class"] = np.where(data["trending_time"]<=3., 1, 0.)
#Data
X = np.array(data_norm[["likes", "dislikes", "views", "comment_count", "trending_time"]])
y = np.array(data["class"]).squeeze()
print(y)
attributeNames = ["likes", "dislikes", "views", "comment_count", "trending_time"]
N, M = X.shape

K = 10
X_train, X_test, y_train, y_test = train_test_split(X, y)

countFast1 = np.count_nonzero(y_train == 1)
baseline_class1 = (countFast1/y_train.size)
print(baseline_class1)

countFast2 = np.count_nonzero(y_test == 1)
baseline_class2 = (countFast2/y_test.size)
print(baseline_class2)

error_test = (( baseline_class2 / baseline_class1)-1)*100
error_test1 = ((baseline_class2-baseline_class1)/(baseline_class1))*100

print(error_test1)


