#!/usr/bin/env python
# coding: utf-8

# In[171]:


import pandas as pd
import numpy as np
import turicreate as tc


# In[172]:


#Read CSV 
beer3 = pd.read_csv('/Users/alisonkamen/desktop/beer3.csv')


# In[173]:


#Create dataframe of required columns then convert to SFrame for turicreate
beer3_1 = beer3[['userId','beer_beerid','review_overall']]
beer3_1 = tc.SFrame(beer3_1)
beer3_1 = beer3_1.dropna()


# In[174]:


#Create SFrame of additional info on beers for model
beer_info = beer3[['beer_beerid','beer_style','beer_abv']].drop_duplicates()
beer_info = tc.SFrame(beer_info)


# In[175]:


#Create training and validation set
training_data, validation_data = tc.recommender.util.random_split_by_user(beer3_1, 'userId', 'beer_beerid')


# In[181]:


#Create item similarity model
beer_model = tc.item_similarity_recommender.create(training_data, 
                                            user_id="userId", 
                                            item_id="beer_beerid", 
                                            item_data=beer_info,
                                            target="review_overall")


# In[177]:


#Save model
beer_model.save("beer_model")


# In[178]:


#Load model
beer_model_load = tc.load_model("beer_model")


# In[127]:


#data2 = tc.SFrame({'userId': [10200,10200,10200],
                          #'beer_beerid': [1550,5441,17538]})


# In[126]:


#model.recommend(['10200'], new_observation_data = data2)


# In[179]:


def user_input():
    input_test = pd.DataFrame(pd.read_json('{"userId": ["101010","101010","101010"], "beer_name": ["IPA","Amber Ale","Founders CBS Imperial Stout"]}')) #JSON input from user
    input_test['beer_beerid'] = pd.DataFrame(beer3.loc[beer3['beer_name'].isin(input_test['beer_name']), 'beer_beerid'].unique()).astype('int64') #Obtain beer id for beer name given by user
    
    predict_frame = tc.SFrame(input_test) #Convert user input dataframe to SFrame
    beer_recs = pd.DataFrame(beer_model.recommend(predict_frame['userId'], new_observation_data = predict_frame)) #Predict new beers for user and convert to dataframe
    
    beer_recs_final = beer_recs.merge(beer3[['beer_name','beer_beerid','beer_abv','beer_style']], on='beer_beerid').drop_duplicates(['beer_beerid']) #Join predictions to beer3 to obtain additional information
    beer_recs_final = beer_recs_final[['beer_name','beer_abv','beer_style']].to_json(orient='records') #Convert desired output to json for iOS output
    
    return beer_recs_final


# In[182]:


user_input()

