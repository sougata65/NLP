from sklearn.externals import joblib
import Sentiment\ Analysis\
import os
import sys
current_path = os.getcwd()
args = sys.argv

data = args[0]
def get_model():
    model = None
    try:
        model = joblib.load(current_path+"/Models/sentiment_analysis_naive_bayes_model.pkl")
    except:
        print("Model not found")

    if model!=None:
        print(model.predict(data))
