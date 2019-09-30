# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 15:12:48 2019

@author: Lukas
"""

#import pandas as pd
#import re
#import numpy as np
import json

def find_cat_labels(json_path):

    # making a dictionary with category labels 
    with open(json_path) as json_file:
        json_data = json.load(json_file)
        #print(d)
        
    cats = dict()

    for i in range(len(json_data["items"])):
        cat = json_data["items"][i]["snippet"]["title"]
        lab = json_data["items"][i]["id"]
        cats[lab] = cat
        #print(json_data["items"][i]["id"])
    
    return cats

#print(find_cat_labels('Datasets/CA_category_id.json'))
#Categories we want to use: 10,15,20,22,26,30



# gives error, but shows all category labels are the same, except nothing is known for US, which is checked manually, do to lack of time
"""
categories = ["CA", "DE", "FR", "GB", "IN", "JP", "KR", "MX", "RU", "US"]

all_files = glob.glob('Datasets/**_category_id.json')



for f in range(len(all_files)):
    with open(all_files[f]) as json_file:
        json_data = json.load(json_file)
    with open(all_files[f+1]) as json_file2:
        json_data2 = json.load(json_file2)

    for i in range(len(json_data["items"])):
       cat1  = json_data["items"][i]["snippet"]["title"]
       cat2  = json_data2["items"][i]["snippet"]["title"]
    
    print(f)
    if cat1 != cat2:
        print(cat1, cat2)
"""    
    
