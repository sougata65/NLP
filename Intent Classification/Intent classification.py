#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import copy


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:


train = pd.read_csv("/Users/sougata-8718/Downloads/train.csv")
test = pd.read_csv("/Users/sougata-8718/Downloads/test_data.csv")


# In[3]:


train = train.append(test.iloc[40], ignore_index = True).drop(['Unnamed: 0'], axis= 1)


# In[4]:


def get_f1_score(predictions, labels, model):
    for average in ["micro", "weighted"]:
        f1 = sklearn.metrics.f1_score(predictions, labels, average = average)
        print("F1 score for ", model.__class__.__name__, " (average = ", average, ") - ", f1)
        pass
    print("\n")
    return f1


# In[5]:


vectorizers = [CountVectorizer(), TfidfVectorizer()]
naive_bayes_models = [GaussianNB, MultinomialNB]
f1_score = sklearn.metrics.f1_score


# In[6]:


for vectorizer in vectorizers:
    print("\nUsing ", vectorizer.__class__.__name__,"\n")
    
    model = SVC(C = 200)
    vectorizer.fit(train["Text"])
    X_train = vectorizer.transform(train["Text"]).toarray()
    X_test = vectorizer.transform(test["Text"]).toarray()
    
    model.fit(X_train, train["Intents"])
    predictions = model.predict(X_test)
    get_f1_score(predictions, test["Intents"], model)
    
    
    
    for naive_bayes_model in naive_bayes_models:
        model = naive_bayes_model()
        model.fit(X_train, train["Intents"])
        predictions = model.predict(X_test)
        get_f1_score(predictions, test["Intents"], model)
        
        pass
    pass


# In[ ]:





# In[ ]:





# In[7]:


# greetings = ['Greetings']
teams = ['Team_leadership_Management', 'Sub_team queries', 'Team_features queries', 'Team_members_count queries', 'Teams_count queries', 'Team_members_information queries', 'Team_members_list queries']
office = ['Dress_code_related queries', 'Office_timing']
doctor = ['Doctor_appointment queries', 'Doctor_availability queries']
dining = ['Dining/mess location related queries', 'Menu related queries']
leave = ['Leave_info', 'Apply_leave procedure']
cust_feedback = ['Complaints', 'Feedback']
misc = ['Greetings','Pitstop_related queries', 'Cab booking', 'Phone_related queries',  'Policies_related queries']

# groups = {'greetings' : greetings, 'teams' : teams, 'office' : office, 'doctor' : doctor,
#         'dining' : dining, 'leave' : leave, 'userfeedback' : cust_feedback, 'misc' : misc}
groups = {'teams' : teams, 'office' : office, 'doctor' : doctor,
        'dining' : dining, 'leave' : leave, 'userfeedback' : cust_feedback, 'misc' : misc}

intent_group_map = {}
for group, intent_list in groups.items():
    for intent in intent_list:
        intent_group_map.update({intent : group})



# In[ ]:





# In[ ]:





# In[8]:


train["group"] = train["Intents"].apply(lambda x : intent_group_map.get(x))
test["group"] = test["Intents"].apply(lambda x : intent_group_map.get(x))
print("Intents sorted into groups")


# In[ ]:





# In[9]:


print("Classification scores for groups\n\n\n")

tfidf_svc_group = None
tfidf_gnb_group = None

for vectorizer in vectorizers:
    print("\nUsing ", vectorizer.__class__.__name__,"\n\n")
    
    X_train = vectorizer.transform(train["Text"]).toarray()
    X_test = vectorizer.transform(test["Text"]).toarray()
    
    model = SVC(C = 200)
    model.fit(X_train, train["group"])
    if vectorizer.__class__.__name__ == "TfidfVectorizer":
        tfidf_svc_group = model    
    predictions = model.predict(X_test)
    
    get_f1_score(predictions, test["group"], model)
    

    
    for naive_bayes_model in naive_bayes_models:
        model = naive_bayes_model()
        model.fit(X_train, train["group"])
        if vectorizer.__class__.__name__ == "TfidfVectorizer" and model.__class__.__name__ == "GaussianNB":
            tfidf_gnb_group = model
        predictions = model.predict(X_test)
        
        get_f1_score(predictions, test["group"], model)
        pass
    pass


# In[ ]:





# In[ ]:





# In[13]:


selected_vectorizers = vectorizers
selected_models = [GaussianNB(), SVC(C = 200)]
group_models = {}

for group, intents in groups.items():
    print("\n\nGroup - ",group)
    X_train_group = train[train["group"] == group]
    X_test_group = test[test["group"] == group]
    
    for vectorizer in selected_vectorizers:
        print("\nUsing ", vectorizer.__class__.__name__,"\n\n")
        #vectorizer.fit(X_train_group["Text"])
        models = []
        for model in selected_models:
            model.fit(vectorizer.transform(X_train_group["Text"]).toarray(), X_train_group["Intents"])
            predictions = model.predict(vectorizer.transform(X_test_group["Text"]).toarray())
            get_f1_score(predictions, X_test_group["Intents"], model)
            models.append(copy.deepcopy(model))
            pass
        if vectorizer.__class__.__name__ == "TfidfVectorizer":
            group_models.update({group : models})


# In[14]:


GaussianNBPredictions = []
SVCPredictions = []


# In[15]:


tfidf_vectorizer = vectorizers[1]
group_classifications_svc = []
group_classifications_gnb = []
def predict_groups(classifier, test_data = test["Text"]):
    return classifier.predict(tfidf_vectorizer.transform(test_data).toarray())

for group_classifier in [tfidf_gnb_group, tfidf_svc_group]:
    if group_classifier.__class__.__name__ == "SVC":
        group_classifications_svc = predict_groups(group_classifier)
        pass
    if group_classifier.__class__.__name__ == "GaussianNB":
        group_classifications_gnb = predict_groups(group_classifier)
    
testCopy = test.copy()

testCopy["group_predictions_gnb"] = group_classifications_gnb
testCopy["group_predictions_svc"] = group_classifications_svc

intent_predictions_gnb_from_gnb = []
intent_predictions_gnb_from_svc = []
intent_predictions_svc_from_gnb = []
intent_predictions_svc_from_svc = []

for idx in range(len(testCopy)):
    group_prediction_gnb = testCopy.iloc[idx]["group_predictions_gnb"]
    group_prediction_svc = testCopy.iloc[idx]["group_predictions_svc"]
    text = testCopy.iloc[idx]["Text"]
    vectorized_text = tfidf_vectorizer.transform([text]).toarray()
    
    for group_prediction in [group_prediction_gnb, group_prediction_svc]:
            
        models = group_models.get(group_prediction)
        
        gnb = models[0]
        svc = models[1]
        
        if group_prediction is group_prediction_gnb:
            intent_predictions_gnb_from_gnb.append(gnb.predict(vectorized_text)[0])
            intent_predictions_svc_from_gnb.append(svc.predict(vectorized_text)[0])
            pass
        if group_prediction is group_prediction_svc:
            intent_predictions_gnb_from_svc.append(gnb.predict(vectorized_text)[0])
            intent_predictions_svc_from_svc.append(svc.predict(vectorized_text)[0])


# In[16]:


testCopy = test.copy()
testCopy["group_gnb_intent_gnb"] = intent_predictions_gnb_from_gnb
testCopy["group_gnb_intent_svc"] = intent_predictions_svc_from_gnb
testCopy["group_svc_intent_gnb"] = intent_predictions_gnb_from_svc
testCopy["group_svc_intent_svc"] = intent_predictions_svc_from_svc


# In[17]:


for column in ["group_gnb_intent_gnb", "group_gnb_intent_svc", "group_svc_intent_gnb", "group_svc_intent_svc"]:
    preds = testCopy[column]
    print('\n\n',column,'\n\n')
    
    get_f1_score(preds, testCopy["Intents"], None)


# In[ ]:





# In[ ]:





# In[ ]:




