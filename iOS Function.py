#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import io
from surprise import KNNBasic, SVD, Reader, Dataset
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


# In[6]:


#Train model
#knn.fit(training_set)
svd.fit(training_set)


# In[7]:


#Create testset from training set..anti testset will predict based off the beers users didnt review
test_set = training_set.build_testset()


# In[8]:


#Predict for each user in the test set - kNN & SVD
#predictions = knn.test(testset3)
predictions = svd.test(test_set)


# In[9]:


#Function to accept user input and recommened new craft beers - user input to be 3 inputs
def user_input(x,y):
    frame = beer3.append({'userId':x,'beer_name':y,'beer_beerid': beer3.loc[beer3['beer_name'] == y, 'beer_beerid']}, ignore_index=True) #Append users beer reviews to dataframe of reviews & find beer id associated with input beer
    frame[['userId','beer_beerid']] = frame[['userId', 'beer_beerid']].apply(lambda x: x.astype(str))
    frame['review_overall'] = frame['review_overall'].astype('float64')
    
    iids = frame['beer_beerid'].unique() #Obtain list of all beer Ids
    iids2 = frame.loc[frame['userId'] == x, 'beer_beerid'] #Obtain list of ids that user has rated
    iids_to_pred = np.setdiff1d(iids,iids2) #List of all beers user didn't rate
                         
    testtest = [[x, beer_beerid, 4.5] for beer_beerid in iids_to_pred] #Array of beers to predict for users      
    predictions2 = pd.DataFrame(svd.test(testtest)) #Predict and convert to DataFrame
    
    predictions2 = predictions2.sort_values(by=['est'], ascending = False)[:5] #Obtain top 5 predictions
    predictions3 = predictions2.merge(beer3[['beer_name','beer_beerid','beer_abv','beer_style']], left_on='iid',right_on='beer_beerid').drop_duplicates(['beer_beerid']) #Join predictions to beer3 to obtain additional information
    
    predictions3 = predictions3[['beer_name','beer_abv','beer_style']].to_dict() #Convert desired output to dictionary for iOS output
    
    return predictions3


# In[10]:


new_recs = user_input('901010','Humulus Lager')


# In[41]:


beer2 = pd.read_json(beer2.to_json(orient='records'))


# In[37]:


#How to read the json into a pandas dataframe
df= pd.read_json(beer2)


# In[28]:


#How to append values from another DF - will use  after our json input is converted to a DF
beer3 = beer3.append(df[df['userId'] == '2'])

