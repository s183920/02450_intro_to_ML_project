# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:48:17 2019

@author: Lukas
"""

from Dataload import make_df
from categories import find_cat_labels
from date_cleaning import date_cleaning
import numpy as np

def clean_data(path):
    df = make_df(path)

    cat_labels = find_cat_labels('Datasets/CA_category_id.json')
    #print(cat_labels)

    df = date_cleaning(df)

    return df

#df = clean_data('Datasets/**videos.csv')

def transform_data(data, cols, standardize = True, log_trans = True):
    """
    Data to transform
    Cols = numeric columns from data to transform
    """
    data = data[cols]
    if log_trans:
        y = np.log(data)
        y[y == -np.inf] = 0
        y[y == np.inf] = 0
        data =  (y- data.mean(axis = 0)) /  data.std(axis = 0)

    if standardize:
        data =  (data- data.mean(axis = 0)) /  data.std(axis = 0)

    return data

"""
cols = ["likes", "dislikes", "views", "comment_count"]
data = clean_data('Datasets/**videos.csv')

data = transform_data(data, cols)
"""

#print("Mean: \n {} \n Std: \n {} \n Head: \n {}".format(data.mean(axis = 0), data.std(axis = 0), data.head()))
data = clean_data('Datasets/**videos.csv')
print(len(data))