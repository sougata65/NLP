from sklearn.externals import joblib
import os
import sys
from train import NaiveBayes

#args = sys.argv
#data = args[0]
class Predictor:
    def __init__(self):
        self.current_path = os.getcwd()

    def get_model(self):
        current_path = self.current_path
        model = None
        vectoriser = None
        try:
            model = joblib.load(current_path+"/Models/sentiment_analysis_naive_bayes_model.pkl")
            vectoriser = joblib.load(current_path+"/Models/vectoriser.pkl")
        except:
            nb = NaiveBayes()
            nb.train_amazon_model()
            model = joblib.load(current_path+"/Models/sentiment_analysis_naive_bayes_model.pkl")
            vectoriser = joblib.load(current_path+"/Models/vectoriser.pkl")
        return model,vectoriser

    def predict(self,string,model,vectoriser):
        list_of_strings = []
        list_of_strings.append(string)
        res = model.predict(vectoriser.transform(list_of_strings).toarray())
        return res
