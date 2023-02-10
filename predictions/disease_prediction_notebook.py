import sys

# coding: utf-8

# # Import libraries

# In[336]:


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score


# In[337]:

def predicted_disease(symptoms, nature_of_disease,sex):
    #Plot the Features Importances



    # # Read the training and testing data
    #

    # In[704]:


    train_df = pd.read_csv("https://raw.githubusercontent.com/mohammad2012191/Drafts/main/data.csv")


    # # Statistics About The Data

    # In[676]:


    train_df.shape


    # In[677]:


    train_df.info()


    # In[678]:


    train_df.head()


    # In[679]:


    train_df.describe()


    # In[680]:


    #The cardinality of each catgorical feature (Training)
    cat_cols = train_df.columns
    for col in cat_cols:
        print(col, train_df[col].nunique())


    # In[687]:


    # # Drop The Missing Values

    # In[705]:


    train_df.dropna(inplace=True)


    # # Feature Engineering

    # ## Put each word in the symptoms as a feature

    # In[706]:


    def Vectorization(data = train_df):
        CouVec = CountVectorizer(max_features=50)
        CouVec.fit(train_df['Symptoms'])
        train_words = CouVec.transform(data['Symptoms']).toarray()
        temp = pd.DataFrame(train_words,columns=CouVec.vocabulary_)
        data = pd.concat([data,temp],axis=1)
        return data


    # ## Label Encoding

    # In[707]:


    def Encoding(data = train_df):
        feats = list(train_df.select_dtypes(include=['object','category']).columns)
        le = LabelEncoder()
        for f in feats:
            le.fit(train_df[f])
            data[f] = le.transform(data[f])
        return data


    # ## Add Aggregations

    # In[708]:


    def Agg(data = train_df, Feature = 'Disease'):
            for feat_1 in ['Symptoms','Nature']:
                data[f'Disease_Agg_{feat_1}_mode'] = data[feat_1].map(dict(train_df.groupby(feat_1)['Disease'].agg(lambda x: pd.Series.mode(x)[0])))
                data[f'Disease_Agg_{feat_1}_nunique'] = data[feat_1].map(dict(train_df.groupby(feat_1)['Disease'].nunique()))
            return data


    # ## Drop Some Useless Features

    # In[709]:


    def DropFeatures(data = train_df):
        data.drop(['Age','Treatment','of','and','in','that'],inplace=True,axis=1)
        return data


    # # Prepare The Dataset

    # In[710]:


    train_df = Vectorization(data=train_df)
    train_df = Encoding(data=train_df)
    train_df = Agg(data=train_df)
    train_df = DropFeatures(data=train_df)


    # # Modeling

    # In[711]:


    lg_params = {'max_depth': 7, 'colsample_bytree':0.6}
    lgbm = LGBMClassifier(**lg_params, random_state=42)


    # ## Validation:

    # In[712]:


    print('Validating...')

    # In[713]:


    # ## Show the Features Importances

    # In[714]:



    # # Inference

    # In[703]:

    data = pd.DataFrame(columns=['Symptoms','Sex','Nature'])
    data.loc[0,:] = [symptoms,sex,nature_of_disease]

    train_df = pd.read_csv("https://raw.githubusercontent.com/mohammad2012191/Drafts/main/data.csv")
    train_df.dropna(inplace=True)
    train_df = pd.concat([train_df,data]).reset_index(drop=True)
    train_df = Vectorization(data=train_df)
    train_df = Encoding(data=train_df)
    train_df = Agg(data=train_df)
    train_df = DropFeatures(data=train_df)

    X = train_df.drop(train_df.shape[0]-1,axis=0).drop('Disease',axis=1).values
    y = train_df.drop(train_df.shape[0]-1,axis=0).loc[:,'Disease'].values

    lgbm.fit(X,y)

    sample = train_df.loc[train_df.shape[0]-1,:].drop('Disease').values.reshape(1,-1)

    Diseases_names = {0: 'Acute Respiratory Distress Syndrome', 1: 'Asbestosis', 2: 'Aspergillosis', 3:
                      'Asthma', 4: 'Bronchiectasis', 5: 'Chronic Bronchitis', 6: 'Influenza', 7: 'Mesothelioma',
                      8: 'Pneumonia', 9: 'Pneumothorax', 10: 'Pulmonary hypertension', 11: 'Respiratory syncytial virus',
                      12: 'Tuberculosis', 13: 'bronchiolitis', 14: 'bronchitis', 15: 'chronic obstructive pulmonary disease',
                      16: 'sleep apnea'}

    return Diseases_names[lgbm.predict(sample)[0]]
