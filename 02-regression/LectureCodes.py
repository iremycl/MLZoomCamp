import numpy as np
import pandas as pd
import os

os.chdir("/Users/iremyucel/MLZoomCamp-1/02-regression")

df = pd.read_csv("data.csv")
df.head()

## Data Preparation
#Inconsistent column names
df.columns.str.lower().str.replace(' ','_')

strings = list(df.dtypes[df.dtypes == "object"].index)

for col in strings:
    df[col] = df[col].str.lower().str.replace(' ','_')

df.head()

## EDA