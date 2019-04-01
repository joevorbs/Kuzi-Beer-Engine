#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from surprise import KNNBasic, SVD, Reader, Dataset
from sklearn.externals import joblib
from collections import defaultdict


# In[2]:


#Read processed csv
beer3 = pd.read_csv('C:/Users/vorbej1/Beer-Engine/beer3.csv')


# In[3]:


#Convert columns to appropriate format
beer3[['userId','beer_beerid']] = beer3[['userId', 'beer_beerid']].apply(lambda x: x.astype(str))


# In[4]:


#Create and prepare training set for model input
reader = Reader(rating_scale=(1, 5))
training_set = Dataset.load_from_df(beer3[['userId', 'beer_beerid', 'review_overall']], reader)
training_set = training_set.build_full_trainset()


# In[5]:


#Set model parameters - kNN & SVD
sim_options = {
    'name': 'pearson_baseline',
    'user_based': True
}
 
knn = KNNBasic(sim_options=sim_options, k=10)
svd = SVD()


# In[14]:


#Train model
#knn.fit(training_set)
svd.fit(training_set)


# In[12]:


#Save Model
joblib.dump(svd, 'recommender_model')


# In[8]:


#Load Model for API
svd_iOS = joblib.load('recommender_model')


# In[9]:


#Function to accept user input and recommened new craft beers
def user_input():
    input_test = pd.DataFrame(pd.read_json('[{"userId": "101010", "beer_name": "Humulus Lager"}]')) #JSON input from user
    input_test['beer_beerid'] = pd.DataFrame(beer3.loc[beer3['beer_name'].isin(input_test['beer_name']), 'beer_beerid'].unique()) #Obtain beer id for beer name given by user
    input_test['userId'] = input_test['userId'].astype(str) #Convert userId column to appropriate format for append
    frame = beer3.append(input_test, sort=True) #Append info to dataframe of all beer reviews 

    frame[['userId','beer_beerid']] = frame[['userId', 'beer_beerid']].apply(lambda x: x.astype(str)) #Convert columns to appropriate format
    frame['review_overall'] = frame['review_overall'].astype('float64')
    
    iids = frame['beer_beerid'].unique() #Obtain list of all beer Ids
    iids2 = frame.loc[frame['userId'][:0], 'beer_beerid'] #Obtain list of ids that user has rated
    iids_to_pred = np.setdiff1d(iids,iids2) #List of all beers user didn't rate
                         
    testtest = [['user', beer_beerid, 4.5] for beer_beerid in iids_to_pred] #Array of beers to predict for users      
    predictions2 = pd.DataFrame(svd.test(testtest)) #Predict and convert to DataFrame
    
    predictions2 = predictions2.sort_values(by=['est'], ascending = False)[:5] #Obtain top 5 predictions
    predictions3 = predictions2.merge(beer3[['beer_name','beer_beerid','beer_abv','beer_style']], left_on='iid',right_on='beer_beerid').drop_duplicates(['beer_beerid']) #Join predictions to beer3 to obtain additional information
    
    predictions3 = predictions3[['beer_name','beer_abv','beer_style']].to_json(orient='index') #Convert desired output to json for iOS output
    
    return predictions3


# In[10]:


result = user_input()


# In[13]:


result

