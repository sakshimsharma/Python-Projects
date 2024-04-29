#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

# Load the data from the uploaded file
file_path = 'train.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe to understand its structure
data.head()


# In[3]:


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load your data
data = pd.read_csv('train.csv')

# Sample the data if it's too large
sample_data = data.sample(n=5000, random_state=42)

# Initialize the TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix_sample = tfidf.fit_transform(sample_data['neighbourhood_group'])

# Initialize and apply the MinMaxScaler
scaler = MinMaxScaler()
numerical_features_sample = scaler.fit_transform(sample_data[['price', 'number_of_reviews', 'reviews_per_month', 'availability_365']])

# Combine TF-IDF results with normalized numerical features
combined_features_sample = np.hstack((tfidf_matrix_sample.toarray(), numerical_features_sample))


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Selecting relevant features
features = ['neighbourhood_group', 'room_type', 'price', 'number_of_reviews', 'reviews_per_month', 'availability_365']

# Handling missing values
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)

# Encoding categorical features using TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['neighbourhood_group'])

# Normalizing numerical features
scaler = MinMaxScaler()
numerical_features = scaler.fit_transform(data[['price', 'number_of_reviews', 'reviews_per_month', 'availability_365']])

# Combine TF-IDF results with normalized numerical features
import numpy as np
combined_features = np.hstack((tfidf_matrix.toarray(), numerical_features))

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(combined_features)

# Display the shape of the cosine similarity matrix to confirm its creation
cosine_sim.shape


# In[13]:


# Taking a smaller sample of the data
sample_data = data.sample(n=5000, random_state=42)

# Reapplying the TF-IDF Vectorizer and normalization
tfidf_matrix_sample = tfidf.transform(sample_data['neighbourhood_group'])
numerical_features_sample = scaler.transform(sample_data[['price', 'number_of_reviews', 'reviews_per_month', 'availability_365']])
combined_features_sample = np.hstack((tfidf_matrix_sample.toarray(), numerical_features_sample))

# Compute the cosine similarity matrix for the sample
cosine_sim_sample = cosine_similarity(combined_features_sample)

# Display the shape of the new cosine similarity matrix
cosine_sim_sample.shape


# In[11]:


def recommend_listings(listing_id, data, cosine_sim, top_n=10):
    
    # Check if the listing ID is in the dataset
    if listing_id not in data['id'].values:
        return "Listing ID not found in the dataset."

    # Get the index of the listing corresponding to the given listing ID
    idx = data.index[data['id'] == listing_id].tolist()[0]

    # Get the pairwise similarity scores of all listings with that listing
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the listings based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the top_n most similar listings
    sim_scores = sim_scores[1:top_n+1]

    # Get the listing indices
    listing_indices = [i[0] for i in sim_scores]

    # Return the top_n most similar listings
    recommended_listings = data.iloc[listing_indices][['id', 'name']]
    recommended_listings['similarity_score'] = [score[1] for score in sim_scores]
    return recommended_listings

# Example usage: Recommend listings similar to a given listing ID
sample_listing_id = sample_data.iloc[0]['id']
recommended_listings = recommend_listings(sample_listing_id, sample_data, cosine_sim_sample)
recommended_listings.head()


# In[ ]:




