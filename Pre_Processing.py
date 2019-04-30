#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#Read Csv
beer = pd.read_csv('C:/Users/Vorbej1/Desktop/beer.csv')


# In[3]:


#Assign unique id to reviewers
beer['userId'] = beer.groupby(['review_profilename']).ngroup()


# In[4]:


#Filter out beers with blank reviews and users with no profile name(user # 1)
beer = beer[(beer['review_overall'] > 0) & (beer['userId'] > 1)]


# In[5]:


#Filter out reviewers who reviewed less than 3 beers
beer = beer.groupby('userId').filter(lambda x: x['userId'].count()>=3)


# In[8]:


#Select relevant columns
beer = beer[['userId','beer_beerid', 'beer_name','brewery_name','beer_style','beer_abv','review_overall','review_time']]


# In[9]:


#Sort so most recent reviews are first when users rated the same beer 2x
beer =  beer.sort_values(by=['userId','beer_beerid', 'beer_name', 'beer_style','beer_abv','review_overall','review_time'], ascending = True)


# In[10]:


#Take first row where user reviewed same beer 2x..most current review will now be taken
beer = beer.groupby(['userId','beer_beerid']).first().reset_index()


# In[11]:


#Identify beers that are given more than one unique ID and keep first
beer_names = pd.DataFrame(beer[['beer_name','beer_beerid']])
beer_names = beer_names.drop_duplicates()
beer_names = beer_names.sort_values(by=['beer_name','beer_beerid'])
beer_names = beer_names.loc[beer_names['beer_name'] != beer_names['beer_name'].shift()]


# In[12]:


#Join now unique ids back to original dataframe so beers only have 1 unique id
beer2 = beer.merge(beer_names, on='beer_beerid')
beer2 = beer2.drop(['beer_name_y'], axis=1)
beer2 = beer2.rename(columns={'beer_name_x' :'beer_name'})


# In[13]:


#Filter out beers that are only rated once
beer2 = beer2.groupby('beer_beerid').filter(lambda x: x['beer_beerid'].count()>1)


# In[14]:


#Write to csv for model and dictionary mapping
beer2.to_csv('beer2.csv', sep=',')


# In[15]:


#Dataframe of unique values for heroku
beer_uniques = beer2.drop_duplicates(subset=['beer_name'])


# In[16]:


#Write to csv - needed for reduced memory usage
beer_uniques.to_csv('beer_uniques.csv', sep=',')

