#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:30:49 2019

@author: Yogeswaran
"""
import resource
import re
import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

list_dir='/home/user/Downloads/aclImdb/'

import time
start_time=time.time()
#Loading Data from local machine
class load_preprocess:
    def __init__(self,list_dir):
        self.list_dir=list_dir
    def load_data(self):
        data_dic={}
        for split in ['train','test']:
            data_dic[split]=[]
            for sentiment in ['pos','neg']:
                score = 1 if sentiment=='pos' else 0
                file_path=os.path.join(self.list_dir,split,sentiment)
                file_names=os.listdir(file_path)
                for file_name in file_names:
                    with open(os.path.join(file_path,file_name),'r') as f:
                        review=f.read()
                        data_dic[split].append([review,score])
        train= pd.DataFrame(data_dic['train'],columns=['text','score'])
        test= pd.DataFrame(data_dic['test'],columns=['text','score'])
        return train,test 
    def clean_text(text):        
        text=re.sub(r'<.*?>','',text)
        text=re.sub(r'\\','',text)
        text=re.sub(r'\'','',text)
        text=re.sub(r'\"','',text).strip().lower()
        filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'    
        text=text.translate(str.maketrans(dict((c,'') for c in filters)))
        return text

#initate Count vectorizer
class _vectorizing:
    def __init__(self,df_train,df_test):
        self.train_data=df_train['text']
        self.test_data=df_test['text']
    def word2vec(self):
        vectorizer=CountVectorizer(stop_words='english',preprocessor=load_preprocess.clean_text)
        training_features=vectorizer.fit_transform(self.train_data)
        test_features=vectorizer.transform(self.test_data)
        return training_features,test_features

#load_data
r=load_preprocess(list_dir)
df_train,df_test=r.load_data()
var=_vectorizing(df_train,df_test)
training_features,test_features=var.word2vec()

#Train Liinear SVC
model=LinearSVC()
model.fit(training_features,df_train['score'])
predicted=model.predict(test_features)

#Evolution 
print(accuracy_score(df_test['score'],predicted))
end_time=time.time()
print(str(end_time-start_time)+' seconds')
print(str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))