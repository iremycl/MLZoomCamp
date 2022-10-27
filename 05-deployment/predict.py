# Load model


#After restarting the kernel
model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in: #w->r or it would overwrite the file
    dv, model = pickle.load(f_in)


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


X = dv.transform([customer])

model.predict_proba(X)[0,1] # Prob that this customer will churn

#Convert all this notebook to single python file that does all these:
#Download as python file

