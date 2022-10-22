import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

os.chdir("/Users/iremyucel/MLZoomCamp-1/04-evaluation")

##4.1 Evaluation metrics

df = pd.read_csv('../03-classification/WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.columns = df.columns.str.lower().str.replace(' ','_')
categorical_columns = list(df.dtypes[df.dtypes=='object'].index)

for c in categorical_columns:
    df[c]=df[c].str.lower().str.replace(' ','_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(0)

df.churn = (df.churn == 'yes').astype(int)

df_full_train, df_test = train_test_split(df, test_size=.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=.25, random_state=1)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values

del df_train['churn']
del df_val['churn']
del df_test['churn']

numerical = ['tenure', 'monthlycharges','totalcharges']

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

dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)
model = LogisticRegression()
model.fit(X_train, y_train)

val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]

churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()

##4.2 Accuracy and dummy model
scores = []
thresholds = np.linspace(0,1,21)
for t in thresholds:
    churn_decision = (y_pred >= t)
    score=accuracy_score(y_val, y_pred >= t)
    #score= (y_val == churn_decision).mean()
    print('%.2f %.3f' % (t,score))
    scores.append(score)

plt.plot(thresholds, scores)

##4.3 Confusion Table
actual_positive = (y_val ==1)
actual_negative = (y_val ==0)

t = .5
predict_positive = (y_pred >=t)
predict_negative = (y_pred <t)

tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()
fn,fp

cofusion_matrix = np.array([
  [tn,fp],
  [fn,tp]  
])

(cofusion_matrix/cofusion_matrix.sum()).round(2)

##4.4 Precision and Recall
# Precision: Fraction of positive examples that are correctly identified
#Positive dediklerinin kaci gercekten pozitif?
#Money you wasted
# Recall: Fraction of correctly identified positive examples
#Positive lerin kacini bulabilmissin?
#Money you didnt spend but shd ve
#1-Recall -> Kacirdigin pozitifler!
r = tp / (tp+fn)

##4.5 ROC Curves

tpr = tp / (tp+fn)
fpr = fp / (fp+tn)
##FPR: Fraction of actual negatives that were incorrectly labelled as positive.
spec = tn/ (fp+tn)
spec
#Specifity: tn / fp+tn : Fraction of actual negatives that were correctly labelled .

#ROC Curve calculates these for all possible thresholds

scores = []
thresholds = np.linspace(0,1,101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)
    
    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()
    
    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    scores.append((t,tp,fp,fn,tn))

columns = ['threshold','tp','fp','fn','tn']
df_scores = pd.DataFrame(scores, columns= columns)
df_scores[::10]
df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
 
plt.plot(df_scores.threshold,df_scores['tpr'], label = 'TPR' )
plt.plot(df_scores.threshold,df_scores['fpr'], label = 'FPR' )
plt.legend()

#Compare it with a model that randomly predicts churning:
np.random.seed(1)
y_rand = np.random.uniform(0,1, size=len(y_val))
((y_rand >= 0.5) == y_val).mean()

def tpr_fpr_dataframe(y_val, y_pred):

    scores = []
    thresholds = np.linspace(0,1,101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()
        scores.append((t,tp,fp,fn,tn))
       
    columns = ['threshold','tp','fp','fn','tn']
    df_scores = pd.DataFrame(scores, columns= columns)
    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
    
    return df_scores

df_rand = tpr_fpr_dataframe(y_val, y_rand)
df_rand[::10]

plt.plot(df_rand.threshold,df_rand['tpr'], label = 'TPR' )
plt.plot(df_rand.threshold,df_rand['fpr'], label = 'FPR' )
plt.legend()

## Ideal s coring model
  
