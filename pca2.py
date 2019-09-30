from clean_data import clean_data
from categories import find_cat_labels
import pandas as pd

df = clean_data('Datasets/**videos.csv')
cat_labels = find_cat_labels('Datasets/CA_category_id.json')

attributeNames = df.columns
#print(cat_labels))
classLabels = [cat_labels[i] for i in ["10","15","20","22","26","30"]] 
print(classLabels)