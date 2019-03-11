#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


#Read Csv
beer1 = pd.read_csv('C:/Users/Vorbej1/Desktop/beer.csv')


# In[3]:


#Subset data 
beer2 = beer1[:100000]


# In[4]:


#Assign unique id to reviewers
beer2['userId'] = beer2.groupby(['review_profilename']).ngroup()


# In[5]:


#Filter out beers with blank reviews and users with no profile name(user # 1)
beer2 = beer2[(beer2['review_overall'] > 0) & (beer2['userId'] > 1)]


# In[6]:


#Select relevant columns
beer2 = beer2[['userId','beer_beerid', 'beer_name', 'beer_style','beer_abv','review_overall','review_time']]


# In[7]:


#Sort so most recent reviews are first when users rated the same beer 2x
beer2 =  beer2.sort_values(by=['userId','beer_beerid', 'beer_name', 'beer_style','beer_abv','review_overall','review_time'], ascending = True)


# In[8]:


#Take first row where user reviewed same beer 2x..most current review will now be taken
beer2 = beer2.groupby(['userId','beer_beerid']).first().reset_index()


# In[9]:


#Identify beers that are given more than one unique ID and keep first
beer_names = pd.DataFrame(beer2[['beer_name','beer_beerid']])
beer_names = beer_names.drop_duplicates()
beer_names = beer_names.sort_values(by=['beer_name','beer_beerid'])
beer_names = beer_names.loc[beer_names['beer_name'] != beer_names['beer_name'].shift()]


# In[10]:


#Join now unique ids back to original dataframe so beers only have 1 unique id
beer3 = beer2.merge(beer_names, on='beer_beerid')
beer3 = beer3.drop(['beer_name_y'], axis=1)
beer3 = beer3.rename(columns={'beer_name_x' :'beer_name'})


# In[11]:


#Convert columns to appropriate format for model & dictionary use
beer3['userId'] = beer3['userId'].astype(str)
beer3['beer_beerid'] = beer3['beer_beerid'].astype(str)
beer3['review_overall'] = beer3['review_overall'].astype('float')


# In[12]:


#Write to csv for model and dictionary mapping
beer3.to_csv('beer3.csv', sep=',')

