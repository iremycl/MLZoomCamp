from statistics import linear_regression
import numpy as np
import pandas as pd
import os

os.chdir("/Users/iremyucel/MLZoomCamp-1/02-regression")

df = pd.read_csv("data.csv")
df.head()

## Data Preparation
#Inconsistent column names
df.columns = df.columns.str.lower().str.replace(' ','_')

#and col values:
string_columns = list(df.dtypes[df.dtypes == 'object'].index)

for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.head()

## EDA

df.dtypes
for col in df.columns:
    print(col)
    print(df[col].unique()[:5])
    print(df[col].nunique())
    
# Price distribution

import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df.msrp[df.msrp < 100000], bins=50)

#The tail would confuse the model, so we need to get rid of it:
np.log([1,10,1000,100000])

np.log1p([0, 1, 10, 1000, 10000])

price_logs= np.log1p(df.msrp)

price_logs

sns.histplot(price_logs, bins=50)
#We had almost normal distribution. better than long tailed

#Missing values
df.isnull().sum()

#Setting up the validation framework
n=len(df)
n_val = int(len(df) * .2)
n_test = int(len(df) * .2)
#n_train  = int(len(df) * .6)
#Cant use the above method because of the rounding.

n_train  = n - n_val - n_test
n_val, n_test, n_train

# df_val = df.iloc[:n_val]
# df_test = df.iloc[n_val:n_val+n_test]
# df_train = df.iloc[n_val+n_test:]
#The problem is that this is sequential, i.e. no bmws in the training data set, we need to shuffle:

idx = np.arange(n)
np.random.seed(2) #to make the results reproducible
np.random.shuffle(idx)

#Get the numbers through this shuffled index:

df_train = df.iloc[idx[:n_train]]
df_val = df.iloc[idx[n_train:n_train+n_val]]
df_test = df.iloc[idx[n_train+n_val:]]

df.iloc[idx[:10]]

len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train.msrp.values)
y_val = np.log1p(df_val.msrp.values)
y_test = np.log1p(df_test.msrp.values)

del df_train['msrp']
del df_val['msrp']
del df_test['msrp']

##Linear Regression

df_train.iloc[10]

xi = [453,11,86]

w0 = 7.17
w = [0.01,0.04,0.002]

def linear_regression(xi):
   n = len(xi)
   pred = w0 
   for j in range(n):
       pred = pred + w[j] * xi[j] # Dot product!!
   return pred

linear_regression(xi)   
np.expm1(linear_regression(xi)) 

##Linear Regression vector form

def dot(xi,w):
    n = len(x1)
    res = 0.0
    for j in range(n):
        res = res + xi[j] * w[j]
    return res

def linear_regression(xi):
   n = len(xi)
   pred = w0 
   for j in range(n):
       pred = pred + w[j] * xi[j] # Dot product!!
   return w0 + dot(xi, w)