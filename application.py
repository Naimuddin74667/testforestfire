import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# importing Linear Regressor & StandardScaler
linear_model = pickle.load(open('Models/Linear_model.pkl','rb'))
standard_scaler = pickle.load(open('Models/scaler.pkl','rb'))

application = Flask(__name__)
app = application

@app.route("/")
def index():
    return render_template('index.html')
@app.route('/predictdata', methods=['GET','POST'])
def predict_data():
    if request.method=='POST':
        temperature = float(request.form.get('Temperature'))  # Corrected to use string literals
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('WS'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        # Transform data and make prediction
        scaled_data = standard_scaler.transform([[temperature, rh, ws, rain, ffmc, dmc, isi, classes, region]])
        result = linear_model.predict(scaled_data)
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')




if __name__=="__main__":
    app.run(host='0.0.0.0',debug=True)
