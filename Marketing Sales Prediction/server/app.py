import csv
import json
import util
import numpy as np
import pandas as pd
# from flask_cors import cross_origin
from werkzeug.utils import secure_filename, redirect
from flask import Flask, render_template, request, jsonify, redirect, send_file
import pickle

# Load the Model back from file

with open("./artifacts/Sales_model.pkl", 'rb') as file:  
    model = pickle.load(file)

app = Flask(__name__)
@app.route("/product_code")
def get_PROD_CD():
    response = jsonify({
        'PROD_CD': util.get_PROD_CD()
    })

    return response
    
@app.route("/salesman_code")
def get_SLSMAN_CD():
    response = jsonify({
        'SLSMAN_CD': util.get_SLSMAN_CD()
    })

    return response
 
@app.route("/predict", methods = ["GET", "POST"])
def predict():
    if request.method == "POST":
        print("Achievment")
        PR = util.get_PROD_CD()
        SL = util.get_SLSMAN_CD()


        PROD_CD = str(request.form["PROD_CD"])#1
        SLSMAN_CD = str(request.form["SLSMAN_CD"])#2
        PLAN_MONTH = request.form["PLAN_MONTH"]#3
        TARGET_IN_EA = request.form["TARGET_IN_EA"]#4
        
        x = np.zeros(len(util.get_columns()))
        x[0] = PR.index(PROD_CD)
        x[1] = SL.index(SLSMAN_CD)
        x[2] = PLAN_MONTH
        x[3] = TARGET_IN_EA
        print(x)

        # x = np.reshape(x,(-1,1))
        prd = model.predict([x])
        #prd = np.reshape(prd,(1,)
        # response = jsonify({
        #     "predictions" : str(np.round(prd[0],4))
        # })
        # response.headers.add('Access-Control-Allow-Origin','*')

        if int(prd)>0:
            return render_template('index.html', prediction_text="Achievment {}".format(np.round(prd.item(), 3)))
        elif int(prd)<0:
            return render_template('index.html', prediction_text="Achievment".format(np.round(-prd.item(),3)))
        # elif int(output)==0:
        #     return render_template('index.html', prediction_text="Achieved.")
        # return response

    return render_template('index.html')

@app.route('/')
def index():
    return render_template("index.html")

if __name__ == "__main__": 
    app.run(debug=True)
