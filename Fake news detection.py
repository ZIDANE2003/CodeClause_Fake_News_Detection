#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Project 1- FAKE NEWS DETECTION


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string


df=pd.read_csv('news.csv')

print(df)


# In[3]:


df


# In[4]:


df.head(5)


# In[5]:


df.tail()


# In[6]:


df.dtypes


# In[7]:


x = df['text']
y = df['label']


# In[8]:


x


# In[9]:


y


# In[10]:


from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[11]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
y_train


# In[12]:


y_train


# In[13]:


tfvect = TfidfVectorizer(stop_words='english',max_df=0.7)
tfid_x_train = tfvect.fit_transform(x_train)
tfid_x_test = tfvect.transform(x_test)


# In[14]:


classifier = PassiveAggressiveClassifier(max_iter=50)
classifier.fit(tfid_x_train,y_train)


# In[15]:


PassiveAggressiveClassifier(C=1.0, average=False, class_weight=None,
                            early_stopping=False, fit_intercept=True,
                            loss='hinge', max_iter=50, n_iter_no_change=5,
                            n_jobs=None, random_state=None, shuffle=True,
                            tol=0.001, validation_fraction=0.1, verbose=0,
                            warm_start=False)


# In[16]:


y_pred = classifier.predict(tfid_x_test)
score = accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[17]:


cf = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print(cf)


# In[18]:


def fake_news_det(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = classifier.predict(vectorized_input_data)
    print(prediction)


# In[19]:


fake_news_det('U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sundayâ€™s unity march against terrorism.')


# In[20]:


fake_news_det('Go to Article Donald Trump was willing to give up a very fulfilling life that took decades to build, so he could step up and take control of an out of control government. He and his family have already sacrificed so much because he chose to put his country first. Making sacrifices is certainly not something loudmouth liberals like Robert DeNiro are accustomed to. DeNiro was very vocal about his opposition to the wildly successful business man Donald J. Trump. He felt so')


# In[21]:


import pickle
pickle.dump(classifier,open('model.pkl', 'wb'))


# In[22]:


# load the model from disk
loaded_model = pickle.load(open('model.pkl', 'rb'))


# In[23]:


def fake_news_det1(news):
    input_data = [news]
    vectorized_input_data = tfvect.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    print(prediction)


# In[24]:


fake_news_det1('By Covert Geopolitics We have been very, very suspicious of Donald Trump since he began his political run.Many believed he was an outsider who was our â€œonly hopeâ€ to tame the US federal government beast. But it has become very clear he is not.')


# In[25]:


fake_news_det1('Hillary Clinton and Bernie Sanders clashed fiercely Sunday over jobs, trade and Wall Street while agreeing that much more must be done to address a two-year-old water-contamination crisis that has paralyzed this majority-black city.')


# In[ ]:




