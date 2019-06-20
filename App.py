from flask import Flask, render_template,request, redirect, url_for, jsonify
import numpy as np
from sklearn.externals import joblib
import predict

app = Flask(__name__)

model = test_model.get_model()


@app.route('/')
def home():
    return render_template("homepage.html")

@app.route('/predict',methods=["POST"])
def predict():
    
    return render_template("result.html")
