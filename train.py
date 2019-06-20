import os
current_path = os.getcwd()
train = open(current_path+"/Data/amazon_reviews_train.csv","r+")
test = open(current_path+"/Data/amazon_reviews_test.csv","r+")
train = train.read()
test = test.read()


# In[ ]:



print("running")

# # Imports

# In[8]:


import numpy as np
import os
import pandas as pd
import nltk
import sklearn
import string
import textblob
from sklearn.externals import joblib
from nltk.stem.snowball import SnowballStemmer,PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB



class NaiveBayes:


    def __init__(self):
        self.replacements = {"can't" : 'can not',"shan't":'shall not',"won't":'will not',"'ve" : " have", "'d" : " would", "'m" : " am", "'ll" : " will", "'s" : "", "n't" : " not","'re" : "are"}
        self.stopwords = set(["can","could","would","have","go","went","zero","one","two","three","four","five","six","seven","eight","nine","ten"]) | set(nltk.corpus.stopwords.words("english"))
        self.stemmer = SnowballStemmer("english")
        print("hooray")
    #defuseful
    def amazon_reviews_text_to_df(self,text):
        documents = text.split("\n")
        df = pd.DataFrame()
        data = []
        labels = []
        for document in documents:
            labels.append(document.split("\t",1)[0])
            text = document.split('\t')[1]
            data.append(text)
        labels = np.array(labels)
        labels[np.where(labels=='__label__2')] = "Positive"
        labels[np.where(labels=='__label__1')] = "Negative"
        df["Data"] = data
        df["Label"] = labels

        return df

    #defuseful
    def remove_punctuation_and_numbers(self,text):
        replacements = self.replacements
        for key,value in replacements.items():
            text = text.replace(key,value)
        text = text.translate(str.maketrans('','',';"#$%&\'()*+/<=>?@[\\]^_`{|}~0123456789')).translate(str.maketrans('!.-:,','     '))
        return text
    #defuseful
    def remove_non_words(self,data):
        res = data.apply(lambda x: self.remove_punctuation_and_numbers(x))
        return res
    #defuseful
    def remove_words_single(self,string,words_to_be_removed):
        words = nltk.word_tokenize(string)
        filtered_words = []
        for i in range(len(words)):
            if words[i] not in words_to_be_removed:
                filtered_words.append(words[i])
        return ' '.join(filtered_words)
    #defuseful
    def remove_words(self,data,words_to_be_removed):
        res = data.apply(lambda x : self.remove_words_single(x,words_to_be_removed))
        return res

    #defuseful
    def stem_single_string(self,string,nltkstemmer):
        words = nltk.word_tokenize(string)
        stemmed_list = []
        for word in words:
            stemmed_list.append(nltkstemmer.stem(word))
        return ' '.join(stemmed_list)

    #defuseful
    def stem(self,data):
        stemmer = SnowballStemmer("english")
        res = data.apply(lambda x : self.stem_single_string(x,stemmer))
        return res
    #defuseful
    def find_rare_words(self,data,max_frequency=4):

        vectoriser = self.get_vectorizer(data)


        temp = ' '.join(data)
        frequencies = (nltk.FreqDist(nltk.word_tokenize(temp)))

        fs = np.array(frequencies.most_common())
        fs = pd.DataFrame(fs)
        fs.columns = ["word","count"]
        fs["freq"] = fs["count"].astype(int)
        fs = fs.drop("count",axis=1)

        rare_words = list(fs[fs["freq"]<=max_frequency]["word"])

        return rare_words
    #defuseful
    def get_vectorizer(self,data,vectorizer="CountVectorizer"):

        if vectorizer == "TFIDF":
            tfidf = TfidfVectorizer()
            tfidf.fit(data)
            print("Using TF-IDF Vectorizer")
            return tfidf
        cv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        cv.fit(data)
        print("Using Count Vectorizer")
        return cv
    #defuseful
    def vectorize_data(self,data,vectorizer="CountVectorizer"):

        if vectorizer == "TFIDF":
            tfidf = TfidfVectorizer()
            tfidf.fit(data)
            print("TF-IDF Vectorizer")
            return tfidf.transform(data).toarray()
        cv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
        cv.fit(data)
        print("Count Vectorizer")
        return cv.transform(data).toarray()

    #defuseful
    def remove_symbols_stopwords_and_stem(self,data):
        data = self.amazon_reviews_text_to_df(data)
        data["Data"] = self.remove_non_words(data["Data"])
        stopwords = self.stopwords
        data["Data"] = self.remove_words(data["Data"],stopwords)
        data["Data"] = self.stem(data["Data"])

        return data

    #defuseful
    def train_amazon_model(self):
        current_path = os.getcwd()
        train = open(current_path+"/Data/amazon_reviews_train.csv","r+")
        test = open(current_path+"/Data/amazon_reviews_test.csv","r+")
        train = train.read()
        test = test.read()

        train = self.remove_symbols_stopwords_and_stem(train)
        test = self.remove_symbols_stopwords_and_stem(test)

        rare_words = self.find_rare_words(train["Data"])

        train["Data"] = self.remove_words(train["Data"],rare_words)
        test["Data"] = self.remove_words(test["Data"],rare_words)

        vectoriser = self.get_vectorizer(train["Data"])

        model = MultinomialNB()
        model.fit(vectoriser.transform(train["Data"]).toarray(),train["Label"])
        test["Prediction"] = model.predict(vectoriser.transform(test["Data"]).toarray())

        joblib.dump(model,current_path+"/Models/sentiment_analysis_naive_bayes_model.pkl")
        joblib.dump(vectoriser,current_path+"/Models/vectoriser.pkl")

        F1_Score = sklearn.metrics.f1_score(np.array(test["Label"])=="Positive",test["Prediction"]=="Positive")
        Accuracy = sklearn.metrics.accuracy_score(test["Label"],test["Prediction"])
        print("Model trained.\n","F1 Score : ",F1_Score,"\nAccuracy : ",Accuracy)
