#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[3]:


movies.head()


# In[4]:


movies.info()


# In[5]:


credits.info()


# In[6]:


credits.head()['cast'].values


# In[7]:


credits.head()['crew'].values


# In[8]:


# now we have to merge both the datsets then we can do it as:

movies=movies.merge(credits,on='title')


# In[9]:


movies.shape


# In[10]:


credits.shape


# In[11]:


# now we want the recommentation system to be a content based so we look all the columns and try to find if it will
# help me in creating the tags for the specific kind of movie or not.

movies.info()


# In[12]:


movies['original_language'].value_counts()

# this is biased towards english only.


# In[13]:


movies['title'].value_counts()


# In[14]:


# genres - tells us about the category of the different movies like comedy, action etc.
# id 
# keywords - like tags
# title
# Overview - summary very important as agar overview sahi h toh baaki do movies match karr sakte hai.


# In[15]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[16]:


movies.info()


# In[17]:


movies.head()


# In[18]:


# the recommendation is based on 3 top cast one the topmost director of the movie and the top 3 crew members 


# # Data Analysis and Cleaning

# In[19]:


# 1. Null Values Data
# 2. Duplicate Data 
# 3. 


# In[20]:


movies.isnull().sum()


# In[21]:


# toh yahan bss 3 null values hai isme 
movies.dropna(inplace=True)


# In[22]:


movies.duplicated().sum()


# In[23]:


movies.iloc[0]['genres']


# In[24]:


# aab upar waale genres ko humme ek list mein laana hai i.e. ['Action','Adventure','Fantasy','SciFi']


# In[ ]:





# In[25]:


import ast # to convert string to a list
def convert(obj):
    l=[]
    
    for i in ast.literal_eval(obj):
        l.append(i['name'])
        
    return l


# In[26]:


print(convert('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'))


# In[27]:


movies['genres']= movies['genres'].apply(convert)


# In[28]:


movies


# In[29]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[30]:


movies['cast'][0]


# In[31]:


def count3(obj):
       l=[]
       count = 0
       
       for i in ast.literal_eval(obj):
           
           if count!=3:
               l.append(i['name'])
               count+=1
           else:
               break
       return l
   


# In[32]:


movies['cast'] = movies['cast'].apply(count3)


# In[33]:


movies


# In[34]:


def fetch_director(obj):
       l=[]
       
       for i in ast.literal_eval(obj):
           
           if i['job'] == 'Director':
               l.append(i['name'])
               break
       return l


# In[35]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[36]:


movies['crew'].unique


# In[37]:


movies


# In[38]:


movies['overview'][0]


# In[39]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[40]:


movies


# In[41]:


# aab yahan ek cheej hai hume spaces hataane padenge ek naam ke andar se warnaa wo same aayenge jaise:
# sam Worthington and Sam Joseph toh tag sirf sam read karegaa

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(' ','') for i in x])


# In[42]:


movies['genres']


# In[43]:


movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(' ','') for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(' ','') for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(' ','') for i in x])


# In[44]:


movies


# In[45]:


movies['tags'] = movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']


# In[46]:


movies


# In[47]:


new_df = movies[['movie_id','title','tags']]


# In[48]:


new_df


# In[49]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[50]:


new_df


# In[51]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[52]:


new_df


# In[53]:


new_df['tags'][0]


# In[54]:


new_df['tags'][1]


# Now we will convert the text into vectors of different orientation and we are going to use the strategy of bag of words

# In[55]:


# aab iske vectors banayenge hum by removing the stopwords from the tags lke to,more etc.

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')


# In[56]:


cv.fit_transform(new_df['tags']).toarray()


# In[57]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[58]:


vectors[0]


# In[59]:


cv.get_feature_names()


# In[60]:


# aab yahan upar bahut saare words hai jo similar hai like accept and accepts toh usse ek vector banalenge hum 
# aab uske liye nlp use karenge hum log 

import nltk


# In[61]:


from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


# In[62]:


# aab ye stemmer words nikalegaa stem words bss toh iske liye ek helper function banalenge

def stem(text):
    
    y=[]
    
    for i in text.split():
        y.append(ps.stem(i))
        
        
    return " ".join(y)


# In[63]:


ps.stem('dancing')


# In[64]:


new_df['tags'][0]


# In[65]:


stem('in the 22nd century, a paraplegic marine is dispatched to the moon pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. action adventure fantasy sciencefiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d samworthington zoesaldana sigourneyweaver jamescameron')


# In[66]:


# toh aab upar isne convert kardiyaa jaise paraplegic ----> parapleg and similarly marine -----> marin


# In[67]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[68]:


new_df


# In[69]:


cv.get_feature_names()


# In[70]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[71]:


vectors.shape


# # we wwould now calculate the angle between the vectors and not the Euclidean Distance between them.
# # Euclidean distance not a reliable measure when dimensions are high, So we are using cosine distance

# In[72]:


from sklearn.metrics.pairwise import cosine_similarity


# In[73]:


similarity = cosine_similarity(vectors)


# In[74]:


similarity


# In[75]:


# toh pehle movie kaa sabb movie ke saath similarity is given as:

similarity[0].shape


# In[76]:


similarity[1]


# In[77]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    # aab agar main sorting karrdungaa then I will loose the index value of it so I will use the enumerate function here.
    movies_list = sorted(list(enumerate(distances)),reverse = True, key=lambda x:x[1])[1:6]
    for item in movies_list:
        print(new_df.iloc[item[0]].title)
        
    


# In[78]:


# toh hum aise index fetch karr sakte hai 
new_df[new_df['title'] == 'Avatar'].index[0]


# In[79]:


# hum aise index saath le sakte hai movie kaa
sorted(list(enumerate(similarity[0])),reverse = True,key = lambda x: x[1])[1:6]


# In[80]:


recommend('Batman Begins')


# In[81]:


# aab agar name fetch karnaa hai then we can use the thing:

new_df.iloc[1216].title

# aab isse recommend function mein paste karr denge hum.


# In[82]:


# aab data yahan se wahan bhejnaa hai apne ko

import pickle


# In[83]:


pickle.dump(new_df, open('movies.pkl','wb'))


# In[84]:


new_df


# In[85]:


new_df['title'].values


# In[86]:


# humne aab iske dictionary banaye hai and we are going to dump this into pickle
new_df.to_dict()


# In[96]:


pickle.dump(new_df.to_dict(), open('movies_dict.pkl','wb'))


# In[ ]:





# In[97]:


# aab similarity matrix ke liye pickle dump karenge hum

pickle.dump(similarity,open('similarity.pkl','wb'))


# In[ ]:




