#!/usr/bin/env python
# coding: utf-8

# In[121]:


import pandas as pd
import numpy as np
import io
from surprise import KNNBasic, SVD, Reader, Dataset
from collections import defaultdict


# In[88]:


#Read processed csv
beer3 = pd.read_csv('C:/Users/vorbej1/Beer-Engine/beer3.csv')


# In[89]:


#Convert columns to appropriate format
beer3[['userId','beer_beerid']] = beer3[['userId', 'beer_beerid']].apply(lambda x: x.astype(str))


# In[90]:


#Create and prepare training set for model input
reader = Reader(rating_scale=(1, 5))
training_set = Dataset.load_from_df(beer3[['userId', 'beer_beerid', 'review_overall']], reader)
training_set = training_set.build_full_trainset()


# In[91]:


#Set model parameters - kNN & SVDD
sim_options = {
    'name': 'pearson_baseline',
    'user_based': True
}
 
knn = KNNBasic(sim_options=sim_options, k=10)
svd = SVD()


# In[92]:


#Train model
#knn.fit(training_set)
svd.fit(training_set)


# In[101]:


joblib.dump(svd, 'model.pkl')


# In[104]:


joe = joblib.load('model.pkl')
joe


# In[122]:


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if joe:
        try:
            json_ = request.json
            print(json_)
            query = pd.get_dummies(pd.DataFrame(json_))
            query = query.reindex(fill_value=0)

            prediction = list(joe.test(query))

            return jsonify({'prediction': str(prediction)})

        except:

            return jsonify({'trace': traceback.format_exc()})
    else:
        print ('Train the model first')
        return ('No model here to use')
    
if __name__ == '__main__':
    try:
        port = int(sys.argv[1]) # This is for a command-line input
    except:
        port = 12345 # If you don't provide any port the port will be set to 12345

    test = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
        #model_columns = joblib.load("model_columns.pkl") # Load "model_columns.pkl"
        #print ('Model columns loaded')

    app.run(port=port, debug=True,use_reloader = False)


# In[93]:


#Create testset from training set..anti testset will predict based off the beers users didnt review
test_set = training_set.build_testset()


# In[94]:


#Predict for each user in the test set - kNN & SVD
#predictions = knn.test(testset3)
predictions = svd.test(test_set)


# In[9]:


#Function to provide top 3 recommendations for each user and output as list
def get_top3_recommendations(predictions, topN = 3):
     
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
     
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key = lambda x:x[0], reverse = False)
        top_recs[uid] = user_ratings[:topN]
     
    return top_recs


# In[10]:


#Map beer ids to beer names using beer3 csv

def read_item_names():
    file_name = ('C:/Users/vorbej1/Beer-Engine/beer3.csv')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split(',')
            rid_to_name[line[2]] = line[3]
            
    return rid_to_name


# In[285]:


#Prints top 3 recommended beers as dictionary values..converts beer id to beer name
top3_recommendations = get_top3_recommendations(predictions)

rid_to_name = read_item_names()

for uid, user_ratings in top3_recommendations.items():
    print (uid, [rid_to_name[iid] for (iid, _) in user_ratings])


# In[12]:


#Predictions on data outside the training set
#beer1 = pd.read_csv('C:/Users/vorbej1/desktop/beer.csv')
#beer1 = beer1.iloc[100001:103000,:]


# In[13]:


#Processing
#beer1['userId'] = beer1.groupby(['review_profilename']).ngroup()
#beer1['userId'] = beer1['userId'].astype('str')
#beer1['beer_beerid'] = beer1['beer_beerid'].astype('str')


# In[14]:


#Create testset
#test_3 = Dataset.load_from_df(beer1[['userId', 'beer_beerid', 'review_overall']], reader)
#test_3 = test_3.build_full_trainset()
#test_3 = test_3.build_testset()


# In[15]:


#Predictions
#predictions2 = svd.test(test_3)
#predictions3 = knn.test(test_3)


# In[96]:


#Function to accept user input and recommened new craft beers - user input to be 3 inputs
def user_input(x,y,z):
    frame = beer3.append({'userId':x,'beer_beerid':y,'review_overall':z}, ignore_index=True) #Append users beer revies to dataframe of reviews
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


# In[97]:


new_recs = user_input('189898','6748', 5)


# In[98]:


new_recs

