from sklearn.externals import joblib
model = joblib.load("sentiment_analysis_naive_bayes_model.pickle")
import sys

args = sys.argv

data = args[0]
model = None
try:
    model = joblib.load("/Models/sentiment_analysis_naive_bayes_model.pkl")
except:
    print("Model not found")

if model!=None:
    print(model.predict(data))
