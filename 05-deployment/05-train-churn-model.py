#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# In[10]:


df = pd.read_csv('../03-classification/WA_Fn-UseC_-Telco-Customer-Churn.csv')

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')
    
df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

df.head(5)
df.shape


# In[11]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

numerical = ['tenure', 'monthlycharges', 'totalcharges']

categorical = [
    'gender',
    'seniorcitizen',
    'partner',
    'dependents',
    'phoneservice',
    'multiplelines',
    'internetservice',
    'onlinesecurity',
    'onlinebackup',
    'deviceprotection',
    'techsupport',
    'streamingtv',
    'streamingmovies',
    'contract',
    'paperlessbilling',
    'paymentmethod',
]

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[12]:


C = 1.0
n_splits = 5

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.churn.values
    y_val = df_val.churn.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[16]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)
y_test = df_test.churn.values
auc = roc_auc_score(y_test, y_pred)
auc


# In[ ]:


#Take this model and put it in a web service


# Save the model

# In[3]:


import pickle


# In[4]:


output_file = f'model_C={C}.bin'
output_file


# In[24]:


f_out = open(output_file,'wb') #write bytes
pickle.dump((dv,model), f_out)
f_out.close()


# In[2]:


with open(output_file, 'wb') as f_out:
    pickle.dump((dv,model),f_out)


# Load model

# In[6]:


#After restarting the kernel
import pickle

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in: #w->r or it would overwrite the file
    dv, model = pickle.load(f_in)


# In[7]:


dv, model


# In[8]:


customer = {
   'gender': 'female',
   'seniorcitizen': 0,
   'partner': 'yes',
   'dependents': 'no',
   'phoneservice': 'no',
   'multiplelines': 'no_phone_service',
   'internetservice': 'dsl',
   'onlinesecurity': 'no',
   'onlinebackup': 'yes',
   'deviceprotection': 'no',
   'techsupport': 'no',
   'streamingtv': 'no',
   'streamingmovies': 'no',
   'contract': 'month-to-month',
   'paperlessbilling': 'yes',
   'paymentmethod': 'electronic_check',
   'tenure': 1,
   'monthlycharges': 29.85,
   'totalcharges': 29.85
}


# In[9]:


X = dv.transform([customer])


# In[11]:


model.predict_proba(X)[0,1] # Prob that this customer will churn


# In[ ]:


#Convert all this notebook to single python file that does all these:
#Download as python file

