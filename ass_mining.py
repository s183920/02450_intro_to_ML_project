from __init__ import binarize2, mat2transactions, print_apriori_rules
import numpy as np
from clean_data import *
from apyori import apriori
import pandas as pd
from categories import find_cat_labels

np.random.seed(180820)

# Load data
cols = ["likes", "dislikes", "views", "comment_count", "trending_time"]
data = clean_data('Datasets/**videos.csv')
data = data[data["category_id"] != 29] # exclude cat 29
cat_labels = find_cat_labels('Datasets/CA_category_id.json')
#cat_col_num = [10,15,20,22] # categories to keep
#data = data[data["category_id"].isin(cat_col_num)]
data_norm = transform_data(data, cols)
#data_norm.head(100)

# find categories
categories = data["category_id"]

# number each category in the data set
cats = {int(k):v for k,v in cat_labels.items()}
cat_count = categories.value_counts().rename(cats)
print(cat_count.to_latex())

"""

# make data frame with one-out-of k encoding of categories
cat_dummy = pd.get_dummies(categories)
cat_dummy.columns = cat_dummy.columns.map(str)
classLabels = [cat_labels[i] for i in cat_dummy.columns]
#cat_dummy.columns = classLabels

# reduce to chosen categories
#cat_cols = ["Music", "Pets & Animals", "Gaming", "People & Blogs"]
cat_cols = classLabels

X = np.array(data_norm[cols])
attributeNames = cols
N, M = X.shape


# We will now transform the wine dataset into a binary format. Notice the changed attribute names:
Xbin, attributeNamesBin = binarize2(X, attributeNames)
cat_dummy = np.array(cat_dummy)

Xbin = np.hstack((Xbin,cat_dummy))
attributeNamesBin = attributeNamesBin+cat_cols

print("X, i.e. the dataset, has now been transformed into:")
print(Xbin)
print(attributeNamesBin)


# Given the processed data in the previous exercise this becomes easy:
T = mat2transactions(Xbin,labels=attributeNamesBin)
#rules = apriori(T, min_support=0.01, min_confidence=0.4) # for category in y itemsets
rules = apriori(T, min_support=0.4, min_confidence=.8) #only outputs the singletons
frules, rules_conf_sup = print_apriori_rules(rules, prints=False)

print("----------------------------------------")
# rename values with upper and lower
rules_conf_sup.replace([' 0th\-50th percentile', ' 50th-100th percentile'],['_lower', '_upper'], regex=True, inplace = True)
# sort by the x itemsets
rules_conf_sup=rules_conf_sup.sort_values(by = "support")
#reset index
rules_conf_sup = rules_conf_sup.reset_index(drop=True)

# category in y itemset
#rules_conf_sup=rules_conf_sup.sort_values(by = "confidence", ascending = False)
#rules_conf_sup = rules_conf_sup[rules_conf_sup["Y itemsets"].isin(cat_cols)].reset_index(drop = True)


with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', None, 'display.max_colwidth', 10**3):  # more options can be specified also
    print(rules_conf_sup.to_latex())
"""