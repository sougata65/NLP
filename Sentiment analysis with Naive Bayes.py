#!/usr/bin/env python
# coding: utf-8

# # Load files

# In[1]:


train = open("/Users/sougata-8718/Downloads/amazon_reviews_train.csv","r+")
test = open("/Users/sougata-8718/Downloads/amazon_reviews_test.csv","r+")
train = train.read()
test = test.read()


# # Imports

# In[2]:


import numpy as np
import pandas as pd
import nltk
import sklearn
import string
import textblob
from nltk.stem.snowball import SnowballStemmer,PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB,MultinomialNB


# # Define necessary functions

# In[3]:


def messy_text_to_df(text):
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

def remove_punctuation_and_numbers(text,replacements):
    for key,value in replacements.items():
        text = text.replace(key,value)
    text = text.translate(str.maketrans('','',';"#$%&\'()*+/<=>?@[\\]^_`{|}~0123456789')).translate(str.maketrans('!.-:,','     '))
    return text
def remove_non_words(data,replacements):
    res = data.apply(lambda x: remove_punctuation_and_numbers(x,replacements))
    return res


def remove_words_single(string,words_to_be_removed):
    words = nltk.word_tokenize(string)
    filtered_words = []
    for i in range(len(words)):
        if words[i] not in words_to_be_removed:
            filtered_words.append(words[i])
    return ' '.join(filtered_words)

def remove_words(data,words_to_be_removed):
    res = data.apply(lambda x : remove_words_single(x,words_to_be_removed))
    return res

    for text,label in documents:
        labels.append(document.split("\t",1)[0])
        text = document.split('\t')[1]
        for key,value in replacements.items():
            text = text.replace(key,value)
            text = text.translate(str.maketrans('','',';"#$%&\'()*+/<=>?@[\\]^_`{|}~0123456789')).translate(str.maketrans('!.-:,','     '))
        words = nltk.word_tokenize(text)
        filtered_words = []
        for i in range(len(words)):
            if words[i] not in words_to_be_removed:
                filtered_words.append(stemmer.stem(words[i]))

        res = ' '.join(filtered_words)
        data.append(res)
    labels = np.array(labels)
    labels[np.where(labels=='__label__2')] = "Positive"
    labels[np.where(labels=='__label__1')] = "Negative"
    return data,labels

def stem_single_string(string,nltkstemmer):
    words = nltk.word_tokenize(string)
    stemmed_list = []
    for word in words:
        stemmed_list.append(nltkstemmer.stem(word))
    return ' '.join(stemmed_list)


def stem(data):
    stemmer = SnowballStemmer("english")
    res = data.apply(lambda x : stem_single_string(x,stemmer))
    return res

def find_rare_words(data,max_frequency=4):

    vectoriser = get_vectorizer(data)


    temp = ' '.join(data)
    frequencies = (nltk.FreqDist(nltk.word_tokenize(temp)))

    fs = np.array(frequencies.most_common())
    fs = pd.DataFrame(fs)
    fs.columns = ["word","count"]
    fs["freq"] = fs["count"].astype(int)
    fs = fs.drop("count",axis=1)

    rare_words = list(fs[fs["freq"]<=max_frequency]["word"])

    return rare_words

def get_vectorizer(data,vectorizer="CountVectorizer"):

    if vectorizer == "TFIDF":
        tfidf = TfidfVectorizer()
        tfidf.fit(data)
        print("TF-IDF Vectorizer")
        return tfidf
    cv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    cv.fit(data)
    print("Count Vectorizer")
    return cv

def vectorize_data(data,vectorizer="CountVectorizer"):
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

    if vectorizer == "TFIDF":
        tfidf = TfidfVectorizer()
        tfidf.fit(data)
        print("TF-IDF Vectorizer")
        return tfidf.transform(data).toarray()
    cv = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
    cv.fit(data)
    print("Count Vectorizer")
    return cv.transform(data).toarray()


# In[4]:


def remove_symbols_stopwords_and_stem(data):
    data = messy_text_to_df(data)
    data["Data"] = remove_non_words(data["Data"],replacements)
    data["Data"] = remove_words(data["Data"],stopwords)
    data["Data"] = stem(data["Data"])

    return data


# # Preprocessing

# In[5]:


replacements = {"can't" : 'can not',"shan't":'shall not',"won't":'will not',"'ve" : " have", "'d" : " would", "'m" : " am", "'ll" : " will", "'s" : "", "n't" : " not","'re" : "are"}
stopwords = set(nltk.corpus.stopwords.words('english'))
stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove("not")
stemmer = SnowballStemmer("english")
stopwords = set(["can","could","would","have","go","went","zero","one","two","three","four","five","six","seven","eight","nine","ten"]) | set(stopwords)

train = remove_symbols_stopwords_and_stem(train)
test = remove_symbols_stopwords_and_stem(test)

rare_words = find_rare_words(train["Data"])

train["Data"] = remove_words(train["Data"],rare_words)
test["Data"] = remove_words(test["Data"],rare_words)


# In[8]:


test


# # Vectorizer

# In[10]:


vectoriser = get_vectorizer(train["Data"])


# # Model Training

# In[11]:


model = MultinomialNB()
model.fit(vectoriser.transform(train["Data"]).toarray(),train["Label"])
test["Prediction"] = model.predict(vectoriser.transform(test["Data"]).toarray())


# # Evaluation

# In[13]:


F1_Score = sklearn.metrics.f1_score(np.array(test["Label"])=="Positive",test["Prediction"]=="Positive")
Accuracy = sklearn.metrics.accuracy_score(test["Label"],test["Prediction"])
print("F1 Score : ",F1_Score,"Accuracy : ",Accuracy)


# In[17]:
