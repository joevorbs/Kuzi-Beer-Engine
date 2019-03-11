#!/usr/bin/env python
# coding: utf-8

# In[125]:


import pandas as pd
import numpy as np
import io
from surprise import KNNBasic, Reader, accuracy, Dataset
from collections import defaultdict


# In[126]:


#Read processed csv
beer3 = pd.read_csv('C:/Users/vorbej1/desktop/beer3.csv')


# In[127]:


#Convert columns to appropriate format
beer3['userId'] = beer3['userId'].astype('str')
beer3['beer_beerid'] = beer3['beer_beerid'].astype('str')


# In[128]:


#Create and prepare training set for model input
reader = Reader(rating_scale=(1, 5))
training_set = Dataset.load_from_df(beer3[['userId', 'beer_beerid', 'review_overall']], reader)
training_set = training_set.build_full_trainset()


# In[129]:


#Set model parameters
sim_options = {
    'name': 'cosine',
    'user_based': True
}
 
knn = KNNBasic(sim_options=sim_options, k=20)


# In[130]:


#Train model
knn.fit(training_set)


# In[131]:


#Create testset from training set..anti testset will predict based off the beers users didnt review
test_set = training_set.build_testset()
#testset2 = training.build_anti_testset()


# In[132]:


#Predict for each user in the test set
predictions = knn.test(test_set)


# In[133]:


#Function to provide top 3 recommendations for each user and output as list
def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x:x[0], reverse = False)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs


# In[134]:


#Map beer ids to beer names using beer3 csv

def read_item_names():
    file_name = ('C:/Users/vorbej1/desktop/beer3.csv')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split(',')
            rid_to_name[line[2]] = line[3]
            
    return rid_to_name


# In[135]:


#Prints top 3 recommended beers as dictionary values..converts beer id to beer name
top3_recommendations = get_top3_recommendations(predictions)

rid_to_name = read_item_names()

for uid, user_ratings in top3_recommendations.items():
    print (uid, [rid_to_name[iid] for (iid, _) in user_ratings])


# In[ ]:


#Predictions on data outside the training set
beer1 = pd.read_csv('C:/Users/vorbej1/desktop/beer.csv')
beer1 = beer1.iloc[100001:103000,:]


# In[ ]:


#Processing
beer1['userId'] = beer1.groupby(['review_profilename']).ngroup()
beer1['userId'] = beer1['userId'].astype('str')
beer1['beer_beerid'] = beer1['beer_beerid'].astype('str')
beer1 = beer1[['userId','beer_beerid','review_overall']]


# In[ ]:


#Create testset
test_3 = Dataset.load_from_df(beer1[['userId', 'beer_beerid', 'review_overall']], reader)
test_3 = test_3.build_full_trainset()
test_3 = test_3.build_testset()


# In[ ]:


#Predictions
predictions2 = knn.test(test_3)

