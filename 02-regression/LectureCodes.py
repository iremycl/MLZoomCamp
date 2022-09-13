import numpy as np
import pandas as pd
import os

os.chdir("/Users/iremyucel/MLZoomCamp-1/02-regression")

df = pd.read_csv("data.csv")
df.head()

#Inconsistent column names
df.columns.str.lower().str.replace(' ','_')
