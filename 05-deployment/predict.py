# Load model
import pickle
from flask import Flask
from flask import request # to get customer info
from flask import jsonify # for JSON files in flask

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in: #w->r or it would overwrite the file
    dv, model = pickle.load(f_in)

app = Flask('churn') #Create a web app
@app.route('/predict', methods=["POST"] ) # As we send some information (in JSON format) about the customer, so POST instead of GET

def predict(): # Turn this into a web service!
    customer = request.get_json() # get customer info and save it as JSON
    
    X = dv.transform([customer])
    y_pred = model.predict_proba(X)[0,1] # Prob that this customer will churn
    churn = y_pred >= 0.5

    result = {
        'churn_probability': float(y_pred), # float turns it into pythonfloat, otherwise its float - float64
        'churn': bool(churn) # bool error, as json does not know how to turn the numpy boolean to text, but it does know how to turn python bool to text, so add bool to return in predict.py
    }
    return jsonify(result) #Response also will be in json.
    
app.run(debug=True, host='0.0.0.0', port=9696)

if __name__=='__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

# WARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.
# For this you can use gunicorn - pip install gunicorn