from flask import Flask, render_template,request, redirect, url_for, jsonify
import numpy as np
from sklearn.externals import joblib
import test_model.py

app = Flask(__name__)

model = test_model.get_model()


@app.route('/')
def home():
    return "good looking homepage"

@app.route('/predict')
def predict():
