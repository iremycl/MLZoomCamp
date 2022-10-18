import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.chdir("/Users/iremyucel/MLZoomCamp-1/03-classification")

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

df.head().T

df.columns = df.columns.str.lower().str.replace(' ','_')
categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

df.head().T
 

df.dtypes
df.totalcharges #shd be numeric
pd.to_numeric(df.totalcharges)
#Error as we replaced all the space with underscore.

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce') #ignore and write as NaN
# we want to replace with 0s.
df.totalcharges = df.totalcharges.fillna(0)

#replace yes/no w/ 1/0
df.churn = (df.churn == 'yes').astype(int)

from sklearn.model_selection import train_test_split
df_full_train, df_test = train_test_split(df, test_size=.2, random_state=1)
len(df_full_train), len(df_test)

#20% of the full data = 25% of the training data after split
df_train, df_val = train_test_split(df_full_train, test_size=.25, random_state=1)
len(df_train), len(df_val), len(df_test)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

## EDA


df_full_train = df_full_train.reset_index(drop=True)
df_full_train.isnull().sum()

df_full_train.churn.value_counts(normalize=True)
global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate,2)

df_full_train.dtypes
numerical=['tenure', 'monthlycharges', 'totalcharges']
df_full_train.columns
categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
    'phoneservice', 'multiplelines', 'internetservice',
    'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
    'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
    'paymentmethod']

df_full_train[categorical].nunique()

#Churn rate
churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_female
churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
churn_male

df_full_train.partner.value_counts()
churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_partner
churn_no_partner =  df_full_train[df_full_train.partner == 'no'].churn.mean()
churn_no_partner

#Perhaps partner var is more important than the gender variable


for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(['mean','count'])
    df_group['diff'] = df_group['mean'] - global_churn_rate
    df_group['risk'] = df_group['mean'] / global_churn_rate
    df_group

#Mutual information - to tell the importance of a categorical variable in prediction. How much do we learn about churn if we observe the ie contract variable?

from sklearn.metrics import mutual_info_score
mutual_info_score(df_full_train.churn, df_full_train.contract)
mutual_info_score(df_full_train.churn, df_full_train.gender)
mutual_info_score(df_full_train.churn, df_full_train.partner)

def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)

mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending = False)

#Correlation - to measure dependency between numerical variables

df_full_train[numerical].corrwith(df_full_train.churn)

df_full_train[df_full_train.tenure <= 2].churn.mean()
df_full_train[df_full_train.tenure > 2].churn.mean()

# One-hot encoding

from sklearn.feature_extraction import DictVectorizer

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False) #Take a dictionary and turn it into a vector
X_train = dv.fit_transform(train_dicts)

#Longer version:
#dv.fit(train_dicts)
#dv.transform(dicts) # To turn it into sparse matrix if you want to
#dv.get_feature_names_out()

#Also for val data
val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

#Logistic Regression

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

z = np.linspace(-5,5,51)
sigmoid(z)

plt.plot(z, sigmoid(z))

def linear_regression(xi):
    result = w0

    for j in range(len(w)):
        result = result + xi[j] * w[j]
    return result

def logistic_regression(xi):
    score = w0 #intermediate score - z

    for j in range(len(w)):
        score = score + xi[j] * w[j]
    result = sigmoid(score)
    return result

# Logistic regression w/ Scikit-Learn

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
model.coef_ # weights, w
model.coef_[0].round(3)

model.intercept_[0] # Bias - intercept

y_pred = model.predict_proba(X_val)[:,1] #2 columns, probability of being 0 and 1. We are interested in it being 1 (col 2), prob of churning.

churn_decision = (y_pred >= 0.5)
df_val[churn_decision].customerid

#Accuracy - Number of correct predictions
(y_val == churn_decision.astype(int)).mean()

df_pred = pd.DataFrame()
df_pred['probability'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
df_pred['correct'] = df_pred.prediction == df_pred.actual
df_pred

# Model Interpretation

dict(zip(dv.get_feature_names(), model.coef_[0].round(3))) #Creates a tuple

small = ['contract', 'tenure', 'monthlycharges']

dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')

dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)
dv_small.get_feature_names()

X_train_small = dv_small.transform(dicts_train_small)
model_small = LogisticRegression()
model_small.fit(X_train_small,y_train)

w0 = model_small.intercept_[0]
w = model_small.coef_[0]
w.round(3)

dict(zip(dv_small.get_feature_names(), w.round(3)))

# Using the model

dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')
dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)
y_full_train = df_full_train.churn.values 

model = LogisticRegression()
model.fit(X_full_train, y_full_train)

dicts_test = df_test[categorical+numerical].to_dict(orient='records')

X_test = dv.transform(dicts_test)

y_pred = model.predict_proba(X_test)[:,1]
churn_decision=(y_pred>=0.5)
(churn_decision == y_test).mean()

customer = dicts_test[-1]
X_small = dv.transform(customer) #Gets a dictionary and creates a numpy array
X_small.shape
model.predict_proba(X_small)[0,1] #Not likely to churn cus40%

