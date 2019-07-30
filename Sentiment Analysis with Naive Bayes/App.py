from flask import Flask, render_template,request, redirect, url_for, jsonify
from flask_restful import Api,Resource,reqparse
import numpy as np
from sklearn.externals import joblib
from Predictor import Predictor
import os
current_path = os.getcwd()
app = Flask(__name__)



@app.route('/')
def home():
    return render_template("homepage.html")

@app.route('/predict',methods=["POST"])
def predict():
    text = request.form["input"]
    predictor = Predictor()
    model,vectoriser = predictor.get_model()
    result = predictor.predict(text,model,vectoriser)
    return render_template("prediction.html",result = result)


app.run(debug=True)
