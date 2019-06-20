from sklearn.externals import joblib
import os
import sys
current_path = os.getcwd()
args = sys.argv

data = args[0]

model = None
vectoriser = None
try:
    model = joblib.load(current_path+"/Models/sentiment_analysis_naive_bayes_model.pkl")
    vectoriser = joblib.load(current_path+"/Models/vectoriser.pkl")
except:
    import train
    model = joblib.load(current_path+"/Models/sentiment_analysis_naive_bayes_model.pkl")
    vectoriser = train.vectoriser
    model = model
    vectoriser = vectoriser

def predict(string):
    list_of_strings = []
    list_of_strings.append(string)
    res = model.predict(vectoriser.transform(list_of_strings).toarray())
    return res
