import os,sys,time,re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from sklearn import grid_search
from sklearn.preprocessing import StandardScaler

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

train_df=pd.read_csv('train_dataset.csv')
test_df=pd.read_csv('test_dataset.csv')

#print(train_df.shape,train_df.head())

def initial_analysis():
    print(train_df.shape, test_df.shape)
    print(train_df.groupby(['Essayset'])['score_1','score_2','score_3','score_4','score_5'].mean())
    print(train_df.groupby(['Essayset','clarity']).size())
    print(train_df.groupby(['Essayset','clarity']).size())
    print(train_df.isnull().sum())

def missing_values():
    #print(train_df.groupby(['Essayset'])['score_1','score_2','score_3','score_4','score_5'].mean())

    for i in range(train_df.shape[0]):
        if train_df.isnull().at[i,'score_3']:
            train_df.at[i,'score_3']=(train_df.at[i,'score_1']+train_df.at[i,'score_2']+train_df.at[i,'score_4']+train_df.at[i,'score_5'])/4.0
        if train_df.isnull().at[i,'score_4']:
            train_df.at[i,'score_4']=(train_df.at[i,'score_1']+train_df.at[i,'score_2']+train_df.at[i,'score_3']+train_df.at[i,'score_5'])/4.0
        if train_df.isnull().at[i,'score_5']:
            train_df.at[i,'score_5']=(train_df.at[i,'score_1']+train_df.at[i,'score_2']+train_df.at[i,'score_3']+train_df.at[i,'score_4'])/4.0


    for i in range(train_df.shape[0]):
        if train_df.isnull().at[i,'Essayset']:
            if train_df.at[i,'max_score'] == 3:
                if (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >= 0 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 <0.28:
                    train_df.at[i,'Essayset']=6.0
                elif (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >=0.28 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 <1.0:
                    train_df.at[i,'Essayset']=5.0
                elif (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >=1.0 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 <1.65:
                    train_df.at[i,'Essayset']=1.0
                elif (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >=1.65 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 <=3:
                    train_df.at[i,'Essayset']=2.0
            elif train_df.at[i,'max_score'] == 2:
                if (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >= 0 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 <0.7:
                    train_df.at[i,'Essayset']=4.0
                elif (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >=0.70 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 <0.9:
                    train_df.at[i,'Essayset']=7.0
                elif (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >=0.9 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 <1.05:
                    train_df.at[i,'Essayset']=3.0
                elif (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >=1.05 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2<1.11:
                    train_df.at[i,'Essayset']=9.0
                elif (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >=1.11 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2<1.15:
                    train_df.at[i,'Essayset']=8.0
                elif (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 >=1.15 and (train_df.at[i,'score_1']+train_df.at[i,'score_2'])/2 <=2.0:
                    train_df.at[i,'Essayset']=10.0

    train_df['avg_score'] = train_df[['score_1', 'score_2', 'score_3', 'score_4', 'score_5']].mean(axis=1)
    train_df['normalized_score'] = train_df['avg_score']/train_df['max_score']
    print(train_df.groupby(['clarity'])['normalized_score'].mean())
    print(train_df.groupby(['coherent'])['normalized_score'].mean())

    for i in range(train_df.shape[0]):
        if train_df.isnull().at[i,'clarity']:
            if train_df.at[i,'normalized_score'] >= 0 and train_df.at[i,'normalized_score'] < 0.24:
                train_df.at[i,'clarity'] = 'worst'
            if train_df.at[i,'normalized_score'] >= 0.24 and train_df.at[i,'normalized_score'] < 0.5:
                train_df.at[i,'clarity'] = 'average'
            if train_df.at[i,'normalized_score'] >= 0.5 and train_df.at[i,'normalized_score'] < 0.79:
                train_df.at[i,'clarity'] = 'above_average'
            if train_df.at[i,'normalized_score'] >= 0.79 and train_df.at[i,'normalized_score'] <= 1.0:
                train_df.at[i,'clarity'] = 'excellent'

    for i in range(train_df.shape[0]):
        if train_df.isnull().at[i,'coherent']:
            if train_df.at[i,'normalized_score'] >= 0 and train_df.at[i,'normalized_score'] < 0.4:
                train_df.at[i,'coherent'] = 'worst'
            if train_df.at[i,'normalized_score'] >= 0.4 and train_df.at[i,'normalized_score'] < 0.7:
                train_df.at[i,'coherent'] = 'average'
            if train_df.at[i,'normalized_score'] >= 0.7 and train_df.at[i,'normalized_score'] < 0.8:
                train_df.at[i,'coherent'] = 'above_average'
            if train_df.at[i,'normalized_score'] >= 0.8 and train_df.at[i,'normalized_score'] <= 1.0:
                train_df.at[i,'coherent'] = 'excellent'

    print(train_df.isnull().sum())
    return train_df   


def normalization():
    pass

def text_process1(textm):
    '''
    Inputs a string of text and performs following activities:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Clean the text and return
    '''
    med=textm.strip()
    med_alpha=re.sub('[^a-zA-Z]',' ',med)
    med_alpha=med_alpha.lower().split()
    med_list=[ps.stem(word) for word in med_alpha if word not in set(stopwords.words('english'))]
    return(" ".join(med_list))

def encoding(data,col):
    data=pd.get_dummies(data=data,columns=col)
    return data

def model1(X_train, X_test, y_train, y_test):
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    y_pred=lr.predict(X_test)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    print("Linear Regression:",rms)

def model2(X_train, X_test, y_train, y_test,test_data):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.fit_transform(X_test)
    parameters = {'kernel': 'linear', 'C': 0.01,'gamma': 1e-7,'epsilon': 0.1}
    svr = SVR(kernel='linear', C=0.01, gamma=1e-7, epsilon=0.1)
    svr.fit(X_train_scaled, y_train)
    y_pred = svr.predict(X_test_scaled)
    rms = sqrt(mean_squared_error(y_test, y_pred))
    print(" SVM Regression:",rms)



if __name__ == "__main__":
    #initial_analysis()
    #train=missing_values()
    #test_df['avg_score'] = test_df[['score_1', 'score_2', 'score_3', 'score_4', 'score_5']].mean(axis=1)
    #test_df['normalized_score'] = test_df['avg_score']/test_df['max_score']
    train=pd.read_csv('after_missing.csv')
    ps=PorterStemmer()
    st_time=time.time()
    train['New Word']=train['EssayText'].astype(str).apply(lambda x: text_process1(x))
    test_df['New Word']=test_df['EssayText'].astype(str).apply(lambda x: text_process1(x))
    print(train['New Word'].head(10))
    bow_transformer=CountVectorizer().fit(train['New Word'])
    messages_bow=bow_transformer.transform(train['New Word'])
    split=int(0.75*messages_bow.shape[0])
    vectorizer = TfidfVectorizer(lowercase=True, max_df=0.9, min_df=2, stop_words='english')
    #vectorizer = TfidfVectorizer()
    #vectorizer = TfidfTransformer()
    tf_idf_matrix = vectorizer.fit_transform(train['New Word']) 
    columns1=vectorizer.get_feature_names()
    tfidf_matrix_test = vectorizer.fit_transform(test_df['New Word']) 
    print("Time duration: ",(time.time()-st_time))
    tfidf_data=pd.DataFrame(tf_idf_matrix.toarray(),columns=columns1)
    total_data=pd.concat([train,tfidf_data], axis=1)
    tfidf_data_test=pd.DataFrame(tfidf_matrix_test.todense())
    total_data_test=pd.concat([test_df,tfidf_data_test], axis=1)
    total_data.drop(['score_1','score_2','score_3','score_4','score_5'], axis=1, inplace=True)
    cols=['Essayset','clarity','coherent']
    total_data=encoding(total_data,cols)
    total_data_test=encoding(total_data_test,cols)
    #X_train = tf_idf_matrix[:split]
    #X_test = tf_idf_matrix[split:]
    #Y_train, Y_test = train['normalized_score'][:split],train['normalized_score'][split:]
    pickle.dump(total_data, open("total_feats.pkl","wb"))
    pickle.dump(total_data_test, open("total_feats_test.pkl","wb"))
    print(total_data.columns[:16])
    print(total_data.shape)
    X_train, X_test, y_train, y_test = train_test_split(total_data.drop(['ID','min_score','max_score','EssayText'],axis=1),total_data['normalized_score'], test_size=0.2)
    model1(X_train, X_test, y_train, y_test)
    model2(X_train, X_test, y_train, y_test)


























