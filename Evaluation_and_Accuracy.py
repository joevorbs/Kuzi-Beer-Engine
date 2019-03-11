#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import io
from surprise import KNNBasic, Reader, accuracy, Dataset
from surprise.model_selection import KFold
from collections import defaultdict


# In[2]:


#Read processed csv
beer3 = pd.read_csv('C:/Users/vorbej1/desktop/beer3.csv')


# In[3]:


#Convert columns to appropriate format
beer3['userId'] = beer3['userId'].astype('str')
beer3['beer_beerid'] = beer3['beer_beerid'].astype('str')


# In[4]:


#Create and prepare training set for model input
reader = Reader(rating_scale=(1, 5))
training_set = Dataset.load_from_df(beer3[['userId', 'beer_beerid', 'review_overall']], reader)
training_set = training_set.build_full_trainset()


# In[5]:


#Set model parameters
sim_options = {
    'name': 'cosine',
    'user_based': True
}
 
knn = KNNBasic(sim_options=sim_options, k=20)


# In[6]:


#Train model
knn.fit(training_set)


# In[ ]:


#Create testset from training set..anti testset will predict based off the beers users didnt review
test_set = training_set.build_testset()
#testset2 = training.build_anti_testset()


# In[ ]:


#Predict for each user in the test set
predictions = knn.test(test_set)


# In[ ]:


#Create dataframe of predictions
predictions_frame = pd.DataFrame(predictions)
predictions_frame['error'] = abs(predictions_frame.est - predictions_frame.r_ui)


# In[ ]:


#Model Accuracy - MAE/RMSE...similar accuracy with item based CF + 10 nn
accuracy.mae(predictions, verbose=True), accuracy.rmse(predictions, verbose=True)


# In[ ]:


#KFold Validation
def precision_recall_at_k(predictions, k=10, threshold=4):

    # First map the predictions to each user.
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    for uid, user_ratings in user_est_true.items():

        # Sort user ratings by estimated value
        user_ratings.sort(key=lambda x: x[0], reverse=True)

        # Number of relevant items
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

        # Number of recommended items in top k
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

        # Number of relevant and recommended items in top k
        n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                              for (est, true_r) in user_ratings[:k])

        # Precision@K: Proportion of recommended items that are relevant
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1

        # Recall@K: Proportion of relevant items that are recommended
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 1

    return precisions, recalls


data = Dataset.load_from_df(beer3[['userId', 'beer_beerid', 'review_overall']], reader)
kf = KFold(n_splits=5)
algo = knn

for trainset, testset in kf.split(data):
    algo.fit(trainset)
    predictions = algo.test(testset)
    precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=4)

    # Precision and recall can then be averaged over all users
    print(sum(prec for prec in precisions.values()) / len(precisions))
    print(sum(rec for rec in recalls.values()) / len(recalls))

