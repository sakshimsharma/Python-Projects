#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# Load and prepare data
data = pd.read_excel('Walmart and S&P 500 .xlsx', parse_dates=["Date"], index_col="Date")
data['S&P 500 Change'] = data['S&P 500'].pct_change().fillna(0)
data['Walmart Change'] = data['WALMART'].pct_change().fillna(0)
data['Performance'] = pd.cut(data['Walmart Change'], bins=[-float('inf'), -0.01, 0.01, float('inf')], labels=['Decrease', 'Stable', 'Increase'])

# Regression analysis
X_reg = sm.add_constant(data['S&P 500'])  # Adding a constant for intercept
y_reg = data['WALMART']
model_reg = sm.OLS(y_reg, X_reg).fit()

# Classification analysis
label_encoder = LabelEncoder()
data['Performance Encoded'] = label_encoder.fit_transform(data['Performance'])
X_class = data[['S&P 500 Change']]
y_class = data['Performance Encoded']
X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.3, random_state=42)
model_class = LogisticRegression(max_iter=200, multi_class='multinomial').fit(X_train, y_train)
y_pred = model_class.predict(X_test)
report = classification_report(y_test, y_pred, target_names=['Decrease', 'Stable', 'Increase'])

# Print results
print(model_reg.summary())
print(report)


# In[3]:


# Load the recently uploaded Excel file to calculate correlation
walmart_sp500_latest_df = pd.read_excel('Walmart and S&P 500 .xlsx')

# Convert 'Date' column to datetime format if necessary and drop any rows with NaN values
walmart_sp500_latest_df['Date'] = pd.to_datetime(walmart_sp500_latest_df['Date'])
walmart_sp500_latest_df.dropna(subset=['WALMART', 'S&P 500'], inplace=True)

# Calculating Pearson correlation coefficient
correlation = walmart_sp500_latest_df['WALMART'].corr(walmart_sp500_latest_df['S&P 500'])

correlation


# In[6]:


import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Load the data from the previously mentioned Excel
data_path = 'Walmart and S&P 500 .xlsx'
wmt_stock_prices = pd.read_excel(data_path, usecols=["Date", "WALMART"], parse_dates=["Date"], index_col="Date")
# Define a function to categorize stock performance
def categorize_performance(change, threshold=0.01):
    if change > threshold:
        return 'Increase'
    elif change < -threshold:
        return 'Decrease'
    else:
        return 'Stable'

# Calculate daily percentage change in the Walmart stock prices
wmt_stock_prices['Daily Change'] = wmt_stock_prices['WALMART'].pct_change()

# Apply categorization function
wmt_stock_prices['Performance'] = wmt_stock_prices['Daily Change'].apply(categorize_performance)

# Dropping initial NaN value from percent change calculation
wmt_stock_prices.dropna(inplace=True)

# Preview the updated dataframe
wmt_stock_prices.head()


# In[7]:


import seaborn as sns

# Counting the occurrences of each performance category
performance_counts = wmt_stock_prices['Performance'].value_counts()

# Plotting the distribution of performance categories
plt.figure(figsize=(8, 5))
sns.barplot(x=performance_counts.index, y=performance_counts.values, palette='coolwarm')
plt.title('Distribution of Stock Performance Categories')
plt.ylabel('Frequency')
plt.xlabel('Performance Category')
plt.show()

performance_counts


# In[8]:


plt.figure(figsize=(14, 7))
plt.plot(data.index, data['WALMART'], label='Walmart Stock Price')
plt.plot(data.index, data['S&P 500'], label='S&P 500 Index', alpha=0.7)
plt.title('Trend Analysis of Walmart Stock Prices and S&P 500')
plt.xlabel('Date')
plt.ylabel('Price/Index Value')
plt.legend()
plt.show()


# In[9]:


#Volatality Changes
# Calculating daily percentage changes
data['Walmart Daily Change'] = data['WALMART'].pct_change() * 100
data['S&P 500 Daily Change'] = data['S&P 500'].pct_change() * 100

# Plotting the volatility
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Walmart Daily Change'], label='Walmart Daily % Change')
plt.plot(data.index, data['S&P 500 Daily Change'], label='S&P 500 Daily % Change', alpha=0.7)
plt.title('Volatility Analysis of Walmart and S&P 500')
plt.xlabel('Date')
plt.ylabel('Daily Percentage Change')
plt.legend()
plt.show()


# In[11]:


#Calculate daily returns
data['Walmart Returns'] = data['WALMART'].pct_change() * 100
data['S&P 500 Returns'] = data['S&P 500'].pct_change() * 100

# Drop any NaN values that arise from the pct_change calculation
data.dropna(inplace=True)

# Correlation analysis
correlation = data['Walmart Returns'].corr(data['S&P 500 Returns'])

# Plotting the correlation of returns
plt.figure(figsize=(10, 6))
sns.scatterplot(x='S&P 500 Returns', y='Walmart Returns', data=data)
plt.title(f'Correlation between Walmart and S&P 500 Returns: {correlation:.2f}')
plt.xlabel('S&P 500 Daily Returns (%)')
plt.ylabel('Walmart Daily Returns (%)')
plt.grid(True)
plt.show()

# Distribution analysis
plt.figure(figsize=(10, 6))
sns.histplot(data['Walmart Returns'], color='blue', kde=True, label='Walmart')
sns.histplot(data['S&P 500 Returns'], color='orange', kde=True, label='S&P 500', alpha=0.7)
plt.title('Distribution of Daily Returns for Walmart and S&P 500')
plt.xlabel('Daily Returns (%)')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[ ]:




