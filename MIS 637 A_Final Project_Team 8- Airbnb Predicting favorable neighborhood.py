#!/usr/bin/env python
# coding: utf-8

# In[1]:


#EXPLORATORY DATA ANALYSIS
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('train.csv')

# Data exploration and preprocessing
# Check for missing values
print(data.isnull().sum())

# Data preprocessing (handling missing values, converting data types if necessary)
data['last_review'] = pd.to_datetime(data['last_review'])

# Analyzing neighborhood trends

# Average price per neighborhood group
avg_price_neighbourhood = data.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)
print("Average price per neighbourhood group:")
print(avg_price_neighbourhood)

# Availability trends per neighborhood group
availability_neighbourhood = data.groupby('neighbourhood_group')['availability_365'].mean().sort_values(ascending=False)
print("Average availability per neighbourhood group:")
print(availability_neighbourhood)

# Reviews per neighborhood group
reviews_neighbourhood = data.groupby('neighbourhood_group')['number_of_reviews'].sum().sort_values(ascending=False)
print("Total reviews per neighbourhood group:")
print(reviews_neighbourhood)

# Visualization of trends
# Plotting average price per neighbourhood group
plt.figure(figsize=(10, 6))
sns.barplot(x=avg_price_neighbourhood.index, y=avg_price_neighbourhood.values)
plt.xlabel('Neighbourhood Group')
plt.ylabel('Average Price')
plt.title('Average Price per Neighbourhood Group')
plt.xticks(rotation=45)
plt.show()

# Plotting average availability per neighbourhood group
plt.figure(figsize=(10, 6))
sns.barplot(x=availability_neighbourhood.index, y=availability_neighbourhood.values)
plt.xlabel('Neighbourhood Group')
plt.ylabel('Average Availability (Days)')
plt.title('Average Availability per Neighbourhood Group')
plt.xticks(rotation=45)
plt.show()

# Plotting total reviews per neighbourhood group
plt.figure(figsize=(10, 6))
sns.barplot(x=reviews_neighbourhood.index, y=reviews_neighbourhood.values)
plt.xlabel('Neighbourhood Group')
plt.ylabel('Total Reviews')
plt.title('Total Reviews per Neighbourhood Group')
plt.xticks(rotation=45)
plt.show()


# In[2]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train.csv')

# Data preprocessing
data['last_review'] = pd.to_datetime(data['last_review'])

# Group data by neighbourhood group and calculate average reviews per month and ratings
neighborhood_stats = data.groupby('neighbourhood_group').agg({
    'reviews_per_month': 'mean',
    'number_of_reviews': 'sum'
}).reset_index()

# Plotting the trends
plt.figure(figsize=(12, 6))

# Plotting average reviews per month by neighborhood
plt.subplot(1, 2, 1)
plt.bar(neighborhood_stats['neighbourhood_group'], neighborhood_stats['reviews_per_month'], color='skyblue')
plt.xlabel('Neighborhood Group')
plt.ylabel('Average Reviews per Month')
plt.title('Average Reviews per Month by Neighborhood Group')
plt.xticks(rotation=45)

# Plotting total number of reviews by neighborhood
plt.subplot(1, 2, 2)
plt.bar(neighborhood_stats['neighbourhood_group'], neighborhood_stats['number_of_reviews'], color='salmon')
plt.xlabel('Neighborhood Group')
plt.ylabel('Total Number of Reviews')
plt.title('Total Number of Reviews by Neighborhood Group')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()



# In[3]:


import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('train.csv')

# Calculate the maximum nights required for booking by neighborhood
max_nights_neighborhood = data.groupby('neighbourhood')['minimum_nights'].max().sort_values(ascending=False)

# Choose the top N neighborhoods to display
top_n = 10  # Change this number to display a different number of neighborhoods

# Get the top N neighborhoods and their corresponding maximum nights required for booking
top_n_neighborhoods = max_nights_neighborhood.head(top_n)

# Plotting
plt.figure(figsize=(10, 6))

# Bar plot for maximum nights required for booking by neighborhood
top_n_neighborhoods.plot(kind='bar', color='salmon')
plt.title(f"Top {top_n} Neighborhoods with Maximum Nights Required for Booking")
plt.xlabel('Neighborhood')
plt.ylabel('Maximum Nights')
plt.xticks(rotation=45, ha='right')  # Rotate labels for better readability
plt.grid(axis='y')

plt.tight_layout()
plt.show()


# In[6]:


#Descriptive Statistics

# Re-importing necessary libraries due to code execution state reset
import pandas as pd

# Attempting to load the newly uploaded dataset for exploratory data analysis
try:
    new_data = pd.read_csv('train.csv')

    # Displaying the first few rows of the dataset to understand its structure
    dataset_preview = new_data.head()
except Exception as e:
    dataset_preview = str(e)

dataset_preview


# In[7]:


#Handling missing values

# Descriptive Statistics of the Dataset
descriptive_stats = new_data.describe()

descriptive_stats


# In[8]:


# Identifying Missing Values in the Dataset
missing_values = new_data.isnull().sum()

missing_values_percentage = (missing_values / len(new_data)) * 100

missing_values, missing_values_percentage


# In[9]:


# Setting up the visualization layout
plt.figure(figsize=(15, 10))

# Distribution of Prices
plt.subplot(2, 2, 1)
sns.histplot(new_data['price'], bins=50, kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.ylabel('Frequency')

# Distribution of Minimum Nights
plt.subplot(2, 2, 2)
sns.histplot(new_data[new_data['minimum_nights'] <= 30]['minimum_nights'], bins=30, kde=True)  # Limiting to 30 nights for better visualization
plt.title('Distribution of Minimum Nights (up to 30 nights)')
plt.xlabel('Minimum Nights')
plt.ylabel('Frequency')

# Distribution of Number of Reviews
plt.subplot(2, 2, 3)
sns.histplot(new_data['number_of_reviews'], bins=50, kde=True)
plt.title('Distribution of Number of Reviews')
plt.xlabel('Number of Reviews')
plt.ylabel('Frequency')

# Availability of Listings Throughout the Year
plt.subplot(2, 2, 4)
sns.histplot(new_data['availability_365'], bins=50, kde=True)
plt.title('Availability of Listings Throughout the Year')
plt.xlabel('Availability (Days)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# In[16]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Selecting only numeric columns for correlation
numeric_data = new_data.select_dtypes(include=[np.number])

# Calculating the correlation matrix for numeric columns only
correlation_matrix = numeric_data.corr()

# Creating a heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features in Airbnb Dataset')
plt.show()


# In[54]:


# Setting up the visualization layout
plt.figure(figsize=(20, 15))

# Price Distribution in Different Neighborhood Groups
plt.subplot(2, 2, 1)
sns.boxplot(x='neighbourhood_group', y='price', data=new_data)
plt.title('Price Distribution in Different Neighborhood Groups')
plt.xlabel('Neighborhood Group')
plt.ylabel('Price')

# Room Type Distribution
plt.subplot(2, 2, 2)
sns.countplot(x='room_type', data=new_data)
plt.title('Room Type Distribution')
plt.xlabel('Room Type')
plt.ylabel('Count')

# Geographical Distribution of Listings
plt.subplot(2, 2, 3)
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group', data=new_data)
plt.title('Geographical Distribution of Listings')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Relationship Between Price and Number of Reviews
plt.subplot(2, 2, 4)
sns.scatterplot(x='price', y='number_of_reviews', data=new_data)
plt.title('Relationship Between Price and Number of Reviews')
plt.xlabel('Price')
plt.ylabel('Number of Reviews')
plt.xlim(0, 1000)  # Limiting x-axis to 1000 for better visibility

plt.tight_layout()
plt.show()


# In[6]:


#USING MACHINE LEARNING ALGORITHM
import pandas as pd

# Load the dataset
file_path = 'train.csv'
data = pd.read_csv(file_path)

# Display the first few rows of the dataframe
data.head()


# In[56]:


# Checking for missing values in the dataset
missing_values = data.isnull().sum()

# Summary statistics for numerical columns
summary_statistics = data.describe()

missing_values, summary_statistics


# In[7]:


# Handling missing values
data['last_review'] = pd.to_datetime(data['last_review'], errors='coerce') # Convert to datetime
data['reviews_per_month'].fillna(0, inplace=True) # Fill missing reviews_per_month with 0
data['name'].fillna('Unknown', inplace=True) # Fill missing names with 'Unknown'
data['host_name'].fillna('Unknown', inplace=True) # Fill missing host names with 'Unknown'

# Re-checking for missing values
missing_values_after = data.isnull().sum()

missing_values_after


# In[13]:


import numpy as np

# Creating aggregated features for the regression task
regression_data = data.groupby('neighbourhood').agg(
    average_price=('price', 'mean'),
    average_minimum_nights=('minimum_nights', 'mean'),
    average_number_of_reviews=('number_of_reviews', 'mean'),
    average_reviews_per_month=('reviews_per_month', 'mean'),
    average_availability_365=('availability_365', 'mean')
).reset_index()

# Preparing the classification labels
# For simplicity, let's consider neighborhoods with average reviews per month greater than the median as 'favorable'
median_reviews_per_month = regression_data['average_reviews_per_month'].median()
classification_data = regression_data.copy()
classification_data['favorable'] = np.where(classification_data['average_reviews_per_month'] > median_reviews_per_month, 1, 0)

regression_data.head(), classification_data.head()


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Regression Task: Predicting the average number of reviews

# Preparing the data
X_regression = regression_data.drop(['neighbourhood', 'average_number_of_reviews'], axis=1)
y_regression = regression_data['average_number_of_reviews']

# Splitting the data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_regression, y_regression, test_size=0.2, random_state=42)

# Initializing and training the Linear Regression model
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)

# Predicting and evaluating the model
y_pred_reg = reg_model.predict(X_test_reg)
regression_mse = mean_squared_error(y_test_reg, y_pred_reg)
regression_r2 = r2_score(y_test_reg, y_pred_reg)

regression_mse, regression_r2


# In[16]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Classification Task: Predicting if a neighborhood is favorable

# Preparing the data
X_classification = classification_data.drop(['neighbourhood', 'favorable'], axis=1)
y_classification = classification_data['favorable']

# Splitting the data
X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(X_classification, y_classification, test_size=0.2, random_state=42)

# Initializing and training the Logistic Regression model
class_model = LogisticRegression()
class_model.fit(X_train_class, y_train_class)

# Predicting and evaluating the model
y_pred_class = class_model.predict(X_test_class)
classification_accuracy = accuracy_score(y_test_class, y_pred_class)
classification_report = classification_report(y_test_class, y_pred_class)

classification_accuracy, classification_report


# In[17]:


# Using the classification model to predict favorable neighborhoods in the existing dataset

# Predicting using the classification model
classification_data['predicted_favorable'] = class_model.predict(X_classification)

# Displaying the predictions along with the neighborhoods
predicted_favorable_neighborhoods = classification_data[['neighbourhood', 'predicted_favorable']]
predicted_favorable_neighborhoods.head()


# In[18]:


# Displaying a larger set of neighborhood predictions
predicted_favorable_neighborhoods.sample(10)  # Randomly selecting 10 rows to show a broader range of predictions


# In[19]:


# Filtering and displaying neighborhood predictions that start with the letter 'B'
neighborhoods_starting_with_b = predicted_favorable_neighborhoods[predicted_favorable_neighborhoods['neighbourhood'].str.startswith('B')]
neighborhoods_starting_with_b


# In[ ]:




