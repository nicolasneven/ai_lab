import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv('ks_projects_2018.csv', sep=',')
# pd.set_option('display.max_rows', None, 'display.max_columns', None)

# print out all available column features
print()
for i in range(len(data.columns.values)):
    print("Col " + str(i) + " - " + data.columns.values[i])
print()

# remove columns/features that are not important to run KNN on
data = data.drop(columns=['ID', 'name', 'category', 'currency', 'deadline', 'goal', 'launched', 'pledged', 'country', 'usd_pledged'])

# drop all rows with missing/null data
data.dropna(inplace=True)

# put 'canceled' and 'suspended' projects into 'failed' category
data = data.replace(['canceled', 'suspended'], 'failed')
# remove 'live' and 'undefined' projects from data set
data = data[data.state != 'live']
data = data[data.state != 'undefined']

# print(data['state'].unique())

# scale numerical data
scaler = MinMaxScaler()
numerical_data = data[['backers', 'usd_pledged_real', 'usd_goal_real']]
scaled_data = scaler.fit_transform(numerical_data)
data[['backers', 'usd_pledged_real', 'usd_goal_real']] = scaled_data

# preview data
print(data)
print()
# print(data.describe())

# one-hot encoding
# print(data['main_category'].unique())
encoder = OneHotEncoder(sparse=False)
qualitative_data = np.array(data['main_category']).reshape(-1, 1)
encoded_data = encoder.fit_transform(qualitative_data)
# print(encoded_data)

labels = data.replace(['failed', 'successful'], [0, 1])
labels = labels['state']
# print(labels.unique())

data = data.drop(columns=['main_category', 'state'])
data[['Publishing', 'Film & Video', 'Music', 'Food', 'Design', 'Crafts', 'Games', 'Comics', 'Fashion', 'Theater', 'Art',
      'Photography', 'Technology', 'Dance', 'Journalism']] = encoded_data
print(data)

# KNN

