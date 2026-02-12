# import the Flask library
from hmac import new
import pickle
import re
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
from regex import D
from sklearn.preprocessing import StandardScaler

# Create the Flask instance and pass the Flask
# constructor, the path of the correct module
application = Flask(__name__)
app=application

## import ridge regressor and StasnderdScaler 
ridge_model = pickle.load(open("models/ridge.pkl",'rb'))
Standerd_scaler = pickle.load(open("models/scaler.pkl",'rb'))


# Default route added using a decorator, for view function 'welcome'
# We pass a simple string to the frontend browser
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=="POST":
        Temperature =float(request.form.get("Temperature"))
        RH= float(request.form.get("RH"))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get("Classes"))
        Region = float(request.form.get('Region'))

        new_data_scaled = Standerd_scaler.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        result = ridge_model.predict(new_data_scaled)

        return render_template("home.html",result=result[0])
    else:
        return render_template("home.html")

# Start with flask web app, with debug as True,# only if this is the starting page
if(__name__ == "__main__"):
    app.run(host="0.0.0.0")