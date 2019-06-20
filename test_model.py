from sklearn.externals import joblib
import os
import sys
current_path = os.getcwd()
args = sys.argv

data = args[0]
model = None
try:
    model = joblib.load(current_path+"/Models/sentiment_analysis_naive_bayes_model.pkl")
except:
    print("Model not found")

if model!=None:
    print(model.predict(data))
