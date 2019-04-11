#!/usr/bin/env python
# coding: utf-8

# In[360]:


import pandas as pd
import numpy as np
import turicreate as tc


# In[361]:


beer3 = pd.read_csv('/Users/alisonkamen/desktop/beer3.csv')


# In[362]:


beer3_1 = beer3[['userId','beer_beerid','review_overall']]


# In[363]:


#beer_data = tc.SFrame.read_csv('/Users/alisonkamen/desktop/beer3.csv')
beer_data = tc.SFrame(beer3_1)
beer_data = beer_data.dropna()


# In[381]:


beer_info = beer3[['beer_beerid','beer_style']]
beer_info = tc.SFrame(beer_info)


# In[382]:


training_data, validation_data = tc.recommender.util.random_split_by_user(beer_data, 'userId', 'beer_beerid')


# In[385]:


model = tc.item_similarity_recommender.create(training_data, 
                                            user_id="userId", 
                                            item_id="beer_beerid", 
                                            item_data=beer_info,
                                            target="review_overall")


# In[376]:


data2 = tc.SFrame({'userId': [10200,10200,10200],
                          'beer_beerid': [41383,46208,46206]})


# In[380]:


model.recommend(['10200'], new_observation_data = data2)

