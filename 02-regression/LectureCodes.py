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
    n = len(xi)
    res = 0.0
    for j in range(n):
        res = res + xi[j] * w[j]
    return res

def linear_regression(xi):
   return w0 + dot(xi, w)

# w = [w0,w1,....,wn] n+1 dim
# xi = [xi0,xi1,xi2,...,xin] xi1 is 1 always:
# wTXi = xiTw = w0 = ..

w_new = [w0] + w
w_new
def linear_regression(xi):
    xi = [1] + xi
    return w0 + dot(xi, w_new)
 
linear_regression(xi)

x1=[1,148,24,1385]
x2=[1,132,24,1385]
x10=[1,453,11,86]
X=[x1,x2,x10]

X=np.array(X)
def linear_regression(X):
    return X.dot(w_new)

#Training a linear regression model
#How do we come with weights?
# (X.T * X)^-1 * (X.T * X) * w = (X.T * X)^-1 * X.T * y

def train_linear_regression(X,y):
    #Find the w vector
    pass

X = [
[148,24,1385],
[132,25,2031],
[453,11,86],
[158,24,185],
[172,25,201],
[413,11,86],
[38, 54,185],
[142,25,431],
[453,31,86]
]

#Bias term - baseline
ones = np.ones(X.shape[0])
ones
X = np.column_stack([ones, X])

y =  [10000,20000,15000,20050,10000,20000,15000,25000,12000]
X=np.array(X)
X
XTX =  X.T.dot(X)
XTX_inv = np.linalg.inv(XTX)
XTX.dot(XTX_inv)

w_full = XTX_inv.dot(X.T).dot(y)
w_full
w0 = w_full[0]
w = w_full[1:]
w0, w