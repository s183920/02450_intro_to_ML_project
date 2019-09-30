# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 14:43:37 2019

@author: gusag
"""

import numpy as np
import pandas as pd 
import glob

def make_df(path):
    all_files = glob.glob(path)
    
    #print(all_files)
    
    li = []
    i = 0
    for filename in all_files: 
        i += 1
        df = pd.read_csv(filename, encoding = 'ISO-8859-1')  
        df = df[['publish_time','trending_date','title','category_id','views','likes','dislikes','comment_count']]
        df['country'] = i
        li.append(df)
    
    frame = pd.concat(li, axis=0, ignore_index=True)
    print("Dataframe made!")
    
    return frame
 

def covariance(Feat1,Feat2,CountryID):
    make_df()
    x = frame[[Feat1,Feat2,'country']]
    x = x[x['country']==CountryID]
    x = x[[Feat1,Feat2]]
    c = np.cov(x.T)
    print(x)
    print(c)

#print(frame)

'''
raw_data = df.get_values()
print(raw_data)

cols = range(0, 7) 
attributeNames = np.asarray(df.columns[cols])
print(attributeNames)
'''
