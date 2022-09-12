import numpy as np
import pandas as pd

df = pd.read_csv("data.csv")
df.head()

#Inconsistent column names
df.columns.str.lower().str.replace(' ','_')
