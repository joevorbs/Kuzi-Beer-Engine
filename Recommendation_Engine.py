#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import io
from surprise import KNNBasic, SVD, Reader, Dataset
from collections import defaultdict


# In[3]:


#Read processed csv
beer3 = pd.read_csv('C:/Users/vorbej1/Beer-Engine/beer3.csv')


# In[5]:


#Convert columns to appropriate format
beer3['userId'] = beer3['userId'].astype('str')
beer3['beer_beerid'] = beer3['beer_beerid'].astype('str')


# In[6]:


#Create and prepare training set for model input
reader = Reader(rating_scale=(1, 5))
training_set = Dataset.load_from_df(beer3[['userId', 'beer_beerid', 'review_overall']], reader)
training_set = training_set.build_full_trainset()


# In[7]:


#Set model parameters - kNN & SVDD
sim_options = {
    'name': 'cosine',
    'user_based': True
}
 
knn = KNNBasic(sim_options=sim_options, k=20)
svd = SVD()


# In[8]:


#Train model
#knn.fit(training_set)
svd.fit(training_set)


# In[9]:


#Create testset from training set..anti testset will predict based off the beers users didnt review
test_set = training_set.build_testset()
#testset2 = training_set.build_anti_testset()


# In[10]:


#Predict for each user in the test set - kNN & SVD
#predictions = knn.test(test_set)
predictions = svd.test(test_set)


# In[11]:


#Function to provide top 3 recommendations for each user and output as list
def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x:x[0], reverse = False)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs


# In[12]:


#Map beer ids to beer names using beer3 csv

def read_item_names():
    file_name = ('C:/Users/vorbej1/Beer-Engine/beer3.csv')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split(',')
            rid_to_name[line[2]] = line[3]
            
    return rid_to_name


# In[2]:


#Prints top 3 recommended beers as dictionary values..converts beer id to beer name
top3_recommendations = get_top3_recommendations(predictions)

rid_to_name = read_item_names()

for uid, user_ratings in top3_recommendations.items():
    print (uid, [rid_to_name[iid] for (iid, _) in user_ratings])


# In[432]:


#Predictions on data outside the training set
#beer1 = pd.read_csv('C:/Users/vorbej1/desktop/beer.csv')
#beer1 = beer1.iloc[100001:103000,:]


# In[433]:


#Processing
#beer1['userId'] = beer1.groupby(['review_profilename']).ngroup()
#beer1['userId'] = beer1['userId'].astype('str')
#beer1['beer_beerid'] = beer1['beer_beerid'].astype('str')


# In[434]:


#Create testset
#test_3 = Dataset.load_from_df(beer1[['userId', 'beer_beerid', 'review_overall']], reader)
#test_3 = test_3.build_full_trainset()
#test_3 = test_3.build_testset()


# In[445]:


#Predictions
#predictions2 = svd.test(test_3)
#predictions3 = knn.test(test_3)


# In[14]:


#Function to accept user input and recommened new craft beers - user input to be 3 inputs
def user_input(x,y,z):
    frame = beer3.append({'userId':x,'beer_beerid':y,'review_overall':z}, ignore_index=True) #Append users beer revies to dataframe of reviews
    frame['userId'] = frame['userId'].astype(str)  #Convert columns to appropriate formats
    frame['beer_beerid'] = frame['beer_beerid'].astype(str)
    frame['review_overall'] = frame['review_overall'].astype('float64')

 
    iids = frame['beer_beerid'].unique() #Obtain list of all beer Ids
    iids2 = frame.loc[frame['userId'] == x, 'beer_beerid'] #Obtain list of ids that user has rated wh
    iids_to_pred = np.setdiff1d(iids,iids2)
                         
    testtest = [[x, beer_beerid, 5] for beer_beerid in iids_to_pred]         
    predictions2 = svd.test(testtest) #Predict
    
    return predictions2


# In[17]:


new_recs = user_input('100000','1550',5)

