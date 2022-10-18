import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mutual_info_score
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("housing.csv")
base = [
'latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income',
'median_house_value',
'ocean_proximity'
]

df = data[base]
df.columns = df.columns.str.lower().str.replace(' ','_')
df.fillna(0, inplace=True)

df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['population_per_household'] = df['population'] / df['households']

#Q1

df.ocean_proximity.mode()

#<1H OCEAN

#Q2

list(df.dtypes[df.dtypes == 'object'].index)

categorical = ['ocean_proximity']
numerical = [
'latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income',
'median_house_value'
]
data_numeric = df.copy()
data_numeric = df[numerical]
data_numeric.corr().unstack().sort_values(ascending = False).head(10)

data_base = df.copy()
#Make median_house_value binary
mean = data_base['median_house_value'].mean()
data_base['above_average'] = np.where(data_base['median_house_value'] >= mean,1,0)
data_base = data_base.drop('median_house_value', axis=1)

#Split data
df_train_full, df_test = train_test_split(data_base, test_size=0.2, random_state=42)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.above_average.values
y_val = df_val.above_average.values
y_test = df_test.above_average.values

#Mutual Info
round(mutual_info_score(df_train.above_average, df_train.ocean_proximity),2)

#Q4

numerical = [
'latitude',
'longitude',
'housing_median_age',
'total_rooms',
'total_bedrooms',
'population',
'households',
'median_income'
]

data_base['ocean_proximity'].value_counts()
train_dict = df_train[numerical + categorical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)
X_train = dv.transform(train_dict)

model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000, random_state=42)
model.fit(X_train, y_train)

#Cal accuracy on validation set
val_dict = df_val[categorical+numerical].to_dict(orient = 'records')
X_val = dv.transform(val_dict)

y_pred = model.predict(X_val)
accuracy = np.round(accuracy_score(y_val, y_pred),2)
print(accuracy)

