# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 08:48:17 2019

@author: Lukas
"""

from Dataload import make_df
from categories import find_cat_labels
from date_cleaning import date_cleaning

def clean_data(path):
    df = make_df(path)

    cat_labels = find_cat_labels('Datasets/CA_category_id.json')
    #print(cat_labels)

    df = date_cleaning(df)

    return df

#df = clean_data('Datasets/**videos.csv')

