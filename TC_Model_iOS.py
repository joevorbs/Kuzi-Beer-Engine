#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import turicreate as tc


# In[2]:


#Read CSV 
beer2 = pd.read_csv('/Users/alisonkamen/desktop/beer2.csv')


# In[7]:


#Create dataframe of required columns then convert to SFrame for turicreate
beer2_1 = beer2[['userId','beer_beerid','review_overall']]
beer2_1 = tc.SFrame(beer2_1)
beer2_1 = beer2_1.dropna()


# In[8]:


#Create SFrame of additional info on beers for model
beer_info = beer2[['beer_beerid','beer_style','beer_abv']].drop_duplicates()
beer_info = tc.SFrame(beer_info)


# In[9]:


#Create training and validation set
training_data, validation_data = tc.recommender.util.random_split_by_user(beer2_1, 'userId', 'beer_beerid')


# In[10]:


#Create item similarity model
beer_model = tc.item_similarity_recommender.create(training_data, 
                                            user_id="userId", 
                                            item_id="beer_beerid", 
                                            item_data=beer_info,
                                            target="review_overall")


# In[11]:


#Save model
beer_model.save("beer_model")


# In[12]:


#Load model
beer_model_load = tc.load_model("beer_model")


# In[127]:


#data2 = tc.SFrame({'userId': [10200,10200,10200],
                          #'beer_beerid': [1550,5441,17538]})


# In[126]:


#model.recommend(['10200'], new_observation_data = data2)


# In[50]:


def user_input():
    input_test = pd.DataFrame(pd.read_json('{"userId": ["101010","101010","101010"], "beer_name": ["Bourbon Chaos","Four O Ice Beer","Krugbier"]}')) #JSON input from user
    input_test['beer_beerid'] = pd.DataFrame(beer2.loc[beer2['beer_name'].isin(input_test['beer_name']), 'beer_beerid'].unique()).astype('int64') #Obtain beer id for beer name given by user
    
    predict_frame = tc.SFrame(input_test) #Convert user input dataframe to SFrame
    beer_recs = pd.DataFrame(beer_model.recommend(predict_frame['userId'], new_observation_data = predict_frame)) #Predict new beers for user and convert to dataframe
    
    beer_recs_final = beer_recs.merge(beer2[['beer_name','beer_beerid','beer_abv','beer_style']], on='beer_beerid').drop_duplicates(['beer_beerid']) #Join predictions to beer3 to obtain additional information
    beer_recs_final = beer_recs_final[['beer_name','beer_abv','beer_style']]#.to_json(orient='records') #Convert desired output to json for iOS output
    
    return beer_recs_final


# In[51]:


result = user_input()


# In[ ]:


result

