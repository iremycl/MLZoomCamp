import numpy as np
import pandas as pd
import os

os.chdir("/Users/iremyucel/MLZoomCamp-1/02-regression")

df = pd.read_csv("data.csv")
df.head()

## Data Preparation
#Inconsistent column names
df.columns = df.columns.str.lower().str.replace(' ','_')

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

df.iloc[n_val]


