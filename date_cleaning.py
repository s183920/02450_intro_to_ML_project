# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:41:01 2019

@author: Lukas
"""
import pandas as pd
#import re
#import numpy as np
#import datetime as dt

#df = frame
#categories = cats

def rearange_date(x):
    b = x.split(sep = "-")
    c = "20"+b[0] +"-" + b[2] + "-" + b[1]
    return c

def date_cleaning(df):
    df["trending_date"] = df.trending_date.replace("\.", "-", regex = True)
    df["trending_date"] = df.trending_date.apply(rearange_date)
    df["trending_date"] = pd.to_datetime(df.trending_date)#.dt.strftime("%Y-%d-%m")
    
    df["publish_time"] = df["publish_time"].replace("T\d{2}:\d{2}:\d{2}\.\d{3}Z", "", regex = True)
    df["publish_time"] = pd.to_datetime(df.publish_time)#.dt.strftime("%Y-%d-%m")
    
    df["trending_time"] = (df["trending_date"]-df["publish_time"]).dt.days
    
    return df


#print(df)