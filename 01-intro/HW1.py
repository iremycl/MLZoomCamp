#Working on VS Code
import pandas as pd
import numpy as np

np.__version__

df = pd.read_csv("data.txt", sep=",")
df.head(10)

df.Make.value_counts().head(3)
df.loc[df['Make']== "Audi"].Model.nunique()
df.isna().sum()

df["Engine Cylinders"].median()
df["Engine Cylinders"].value_counts().head(n=10)
df["Engine Cylinders"] = df["Engine Cylinders"].fillna(df["Engine Cylinders"].mode()[0])
df["Engine Cylinders"].median()
df.isna().sum()

X = df.loc[df['Make']=="Lotus",["Engine HP", "Engine Cylinders"]].drop_duplicates().to_numpy()
XTX = np.matmul(X.T,X)
iXTX=np.linalg.inv(XTX)
y=np.array([1100, 800, 750, 850, 1300, 1000, 1000, 1300, 800])
A = np.matmul(iXTX,X.T)
w=A.dot(y)
w[0]
