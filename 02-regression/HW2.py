from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.chdir("/Users/iremyucel/MLZoomCamp-1/02-regression")

data = pd.read_csv("housing.csv")
data.columns
base = [
'longitude', 
'latitude', 
'housing_median_age', 
'total_rooms',
'total_bedrooms', 
'population', 
'households', 
'median_income',
'median_house_value',
]

df_base = data[base]
df_base
for f in base:
    print(f, data[f].isna().sum())

df_base['population'].median()

n=len(df_base)

n_val = int(len(df_base) * .2)
n_test = int(len(df_base) * .2)
n_train  = n - n_val - n_test
n_val, n_test, n_train

idx = np.arange(n)
np.random.seed(42)
np.random.shuffle(idx)

df_train = df_base.iloc[idx[:n_train]]
df_val = df_base.iloc[idx[n_train:n_train+n_val]]
df_test = df_base.iloc[idx[n_train+n_val:]]

len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train['median_house_value'].values)
y_val = np.log1p(df_val['median_house_value'].values)
y_test = np.log1p(df_test['median_house_value'].values)

del df_train['median_house_value']
del df_val['median_house_value']
del df_test['median_house_value']

X_train = df_train.values
X_val = df_val.values
X_test = df_test.values

df_w0 = df_train.copy()
df_w0['total_bedrooms'] = df_w0['total_bedrooms'].fillna(0)
df_w0['total_bedrooms'].isna().sum()
df_w0.mean()
X_train_w0 =df_w0.values

df_wmean = df_train.copy()
df_wmean['total_bedrooms'] = df_wmean['total_bedrooms'].fillna(df_train['total_bedrooms'].mean())
df_wmean['total_bedrooms'].isna().sum()
df_wmean.mean()
X_train_wmean =df_wmean.values

def train_linear_regression(X,y):
    #Find the w vector
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    XTX =  X.T.dot(X)
    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]

def rmse(y,y_pred):
    se = (y - y_pred) ** 2
    mse = se.mean()
    return np.sqrt(mse)

w00, w0 = train_linear_regression(X_train_w0, y_train)
w0m, wm = train_linear_regression(X_train_wmean, y_train)

y_pred0 = w00 + X_train_w0.dot(w0)
y_predm = w0m + X_train_wmean.dot(wm)

score0 = rmse(y_train, y_pred0)
scoreMean = rmse(y_train, y_predm)
print(score0.round(2), scoreMean.round(2))
# 0.34 0.34


def train_linear_regression_reg(X,y,r=0.001):
    #Find the w vector
    ones = np.ones(X.shape[0])
    X = np.column_stack([ones, X])
    
    XTX = X.T.dot(X)
    XTX = XTX + r * np.eye(XTX.shape[0])

    XTX_inv = np.linalg.inv(XTX)
    w_full = XTX_inv.dot(X.T).dot(y)

    return w_full[0], w_full[1:]


rs = [0, 0.000001, 0.0001, 0.001, 0.01, 0.1, 1, 5, 10]
for r in rs:
    w_0, w = train_linear_regression_reg(X_train_w0, y_train, r=r)
    y_pred = w_0 + X_val.dot(w)
    score = rmse(y_val, y_pred)
    score = round(score, 2)
    print(r, w_0, score)


#Q5

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
scores = [] #Empty list to store the scores
for seed in seeds:
    df6 = df_base.copy()
    df6['total_bedrooms'] = df6['total_bedrooms'].fillna(0)

    n=len(df6)
    n_val = int(len(df6) * .2)
    n_test = int(len(df6) * .2)
    n_train  = n - n_val - n_test
    #n_val, n_test, n_train
    
    np.random.seed(seed)
    idx = np.arange(n)
    np.random.shuffle(idx)

    df6_train = df6.iloc[idx[:n_train]]
    df6_val = df6.iloc[idx[n_train:n_train+n_val]]
    df6_test = df6.iloc[idx[n_train+n_val:]]

    #len(df6_train), len(df6_val), len(df6_test)

    df6_train = df6_train.reset_index(drop=True)
    df6_val = df6_val.reset_index(drop=True)
    df6_test = df6_test.reset_index(drop=True)

    y_train = np.log1p(df6_train['median_house_value'].values)
    y_val = np.log1p(df6_val['median_house_value'].values)
    y_test = np.log1p(df6_test['median_house_value'].values)

    del df6_train['median_house_value']
    del df6_val['median_house_value']
    del df6_test['median_house_value']

    X_train = df6_train.values
    X_val = df6_val.values
    X_test = df6_test.values

    w00, w0 = train_linear_regression(X_train, y_train)
    y_pred0 = w00 + X_val.dot(w0)
    score0 = rmse(y_val, y_pred0)
    scores.append(score0)
    print(seed, score0)

scores = np.array(scores)
scores
np.std(scores).round(3)

# 0.00417

# Q6

np.random.seed(9)
idx = np.arange(n)
np.random.shuffle(idx)

n_test = int(len(df_base) * .2)
n_train  = n  - n_test
n_test, n_train

df_train = df_base.iloc[idx[:n_train]]
df_test = df_base.iloc[idx[n_train:]]

len(df_train), len(df_test)

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = np.log1p(df_train['median_house_value'].values)
y_test = np.log1p(df_test['median_house_value'].values)

del df_train['median_house_value']
del df_test['median_house_value']

X_train = df_train.values
X_test = df_test.values

df_train['total_bedrooms'] = df_train['total_bedrooms'].fillna(0)
df_train['total_bedrooms'].isna().sum()
df_train.mean()
X_train =df_train.values


w0, w = train_linear_regression(X_train, y_train)

y_pred = w0 + X_train.dot(w)

score0 = rmse(y_train, y_pred)
print(score0.round(4))