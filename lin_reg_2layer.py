from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, title, subplot, show, grid)
import numpy as np
from scipy.io import loadmat
import sklearn.linear_model as lm
from sklearn import model_selection
from __init__ import rlr_validate
from clean_data import clean_data, transform_data

cols = ["likes", "dislikes", "views", "comment_count", "trending_time"]
data = clean_data('Datasets/**videos.csv')
data_norm = transform_data(data, cols)


def linear_regression_CV(data, X_cols, y_col):
    """
    X_cols: name of cols to use for prediction
    y_col: column to predict
    """
    X = np.array(data[X_cols])
    y = np.array(data[y_col]).squeeze()

    #attributeNames = [name[0] for name in colnames]
    attributeNames = X_cols
    N, M = X.shape

linear_regression_CV(data_norm, X_cols = ["likes", "dislikes", "comment_count", "trending_time"], y_col = "views")