from statistics import linear_regression
from unicodedata import category
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
X = np.column_stack([ones, X])

y =  [10000,20000,15000,20050,10000,20000,15000,25000,12000]

def train_linear_regression(X,y):
    #Find the w vector
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX =  X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0],w_full[1:]


train_linear_regression(X,y)

# Car price baseline model
df_train.dtypes
df_train.columns
base = [
    'engine_hp',
    'engine_cylinders',
    'highway_mpg',
    'city_mpg',
    'popularity'
]

X_train = df_train[base].values # to extract the numpy array
train_linear_regression(X_train, y_train) # nan because of missing values

X_train = df_train[base]. fillna(0).values # doesnt make sense, if for example the nan is in engine horsepower or cylinder, a car cant have 0 cylinders but easy to implement in this case
w0, w = train_linear_regression(X_train, y_train)

#Use to predict

y_pred = w0 + X_train.dot(w)

sns.histplot(y_pred, color = 'red', alpha=0.5, bins=50)
sns.histplot(y_train, color = 'blue', alpha=0.5, bins=50)

#Evaluating regression models - root mean squared error

def rmse(y,y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

rmse(y_train, y_pred)

#Use on validation

base = [
    'engine_hp',
    'engine_cylinders',
    'highway_mpg',
    'city_mpg',
    'popularity'
]

w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)

def prepare_X(df):
    df_num = df[base]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

# Improve the model!

df_train # year is an important value!

def prepare_X(df):
    df = df.copy()
    df['age'] = 2017 - df.year
    features = base + ['age']

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
X_train # has 6 features ones, last is age

w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
rmse(y_val, y_pred)

sns.histplot(y_pred, color = 'red', alpha=0.5, bins=50)
sns.histplot(y_val, color = 'blue', alpha=0.5, bins=50)

# Categorical variables

df_train.number_of_doors # looks numerical but its actually categorical

def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')
    
    for v in [2,3,4]:
        df['num_doors_%s' % v] = (df.number_of_doors == v).astype('int')
        features.append('num_doors_%s' % v)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X


X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred) # Slightly improved, the number of doors is not that useful.


makes = list(df.make.value_counts().head().index)

def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')
    
    for v in [2,3,4]:
        df['num_doors_%s' % v] = (df.number_of_doors == v).astype('int')
        features.append('num_doors_%s' % v)

    for v in makes:
        df['make_%s' % v] = (df.make == v).astype('int')
        features.append('make_%s' % v)

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)
# Add more categorical variables
df_train.dtypes

categorical_variables = [
    'make',
    'engine_fuel_type',
    'transmission_type',
    'driven_wheels',
    'market_category',
    'vehicle_size',
    'vehicle_style'
]

categories = {}

for c in categorical_variables:
    categories[c] = list(df[c].value_counts().head().index)


def prepare_X(df):
    df = df.copy()
    features = base.copy()

    df['age'] = 2017 - df.year
    features.append('age')
    
    for v in [2,3,4]:
        df['num_doors_%s' % v] = (df.number_of_doors == v).astype('int')
        features.append('num_doors_%s' % v)
    
    for c, values in categories.items():
        for v in values:
            df['%s_%s' % (c, v)] = (df[c] == v).astype('int')
            features.append('%s_%s' % (c, v))

    df_num = df[features]
    df_num = df_num.fillna(0)
    X = df_num.values
    
    return X

X_train = prepare_X(df_train)
w0, w = train_linear_regression(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred) #Made the model worse!

#There could be duplicate - or almost duplicate columns in our data. We shouldnt be able to compute the inverse of it but since they are not exactly duplicates, python can and the results are huge numbers.
# Add a small number to diagonal, to make it less possible of duplicate columns

#Regularization - control w so they dont grow too much.

def train_linear_regression_reg(X,y,r=0.001):
    #Find the w vector
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


X_train = prepare_X(df_train)
w0, w = train_linear_regression_reg(X_train, y_train)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)

rmse(y_val, y_pred)

#FIND THE BEST REGULARIZATION PARAMETER

for r in [0.0, 0.00001, 0.0001, 0.001, 0.1, 1, 10]:
    X_train = prepare_X(df_train)
    w_0, w = train_linear_regression_reg(X_train, y_train, r=r)

    X_val = prepare_X(df_val)
    y_pred = w0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    print(r, w0, score)

r=0.001
X_train = prepare_X(df_train)
w_0, w = train_linear_regression_reg(X_train, y_train, r=r)

X_val = prepare_X(df_val)
y_pred = w0 + X_val.dot(w)
score = rmse(y_val, y_pred)
score

# Use the model
df_full_train = pd.concat([df_train, df_val])
df_full_train = df_full_train.reset_index(drop=True)

X_full_train = prepare_X(df_full_train)

X_full_train

y_full_train = np.concatenate([y_train, y_val])
w_0, w = train_linear_regression_reg(X_full_train, y_full_train, r=0.001)


X_test = prepare_X(df_test)
y_pred = w0 + X_test.dot(w)
score = rmse(y_test, y_pred)
score

car = df_test.iloc[20].to_dict()

df_small = pd.DataFrame([car])
X_small = prepare_X(df_small)
y_pred = w0 + X_small.dot(w)
y_pred = y_pred[0]

np.expm1(y_pred)
np.expm1(y_test[20])