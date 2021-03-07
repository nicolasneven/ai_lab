import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('ks_projects_2018.csv', sep=',')
pd.set_option('display.max_rows', None, 'display.max_columns', None)

# print out all available column features
"""
print()
for i in range(len(data.columns.values)):
    print("Col " + str(i) + " - " + data.columns.values[i])
print()
"""


# remove columns/features that are not important to run KNN on
data = data.drop(columns=['ID', 'name', 'category', 'currency', 'deadline', 'goal', 'launched', 'pledged', 'country',
                          'usd_pledged'])

# drop all rows with missing/null data
data.dropna(inplace=True)

# put 'canceled' and 'suspended' projects into 'failed' category
data = data.replace(['canceled', 'suspended'], 'failed')
# remove 'live' and 'undefined' projects from data set
data = data[data.state != 'live']
data = data[data.state != 'undefined']

# preview data
# print(data['state'].unique())
print("\nKICKSTARTER DATA")
print(data.head())
print("\nDATA SUMMARY")
print(data.describe())

# scale numerical data
scaler = MinMaxScaler()
numerical_data = data[['backers', 'usd_pledged_real', 'usd_goal_real']]
scaled_data = scaler.fit_transform(numerical_data)
data[['backers', 'usd_pledged_real', 'usd_goal_real']] = scaled_data

# preview data
# print(data.head())
# print()
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
# print(data.head())

# KNN
train_size = 8000
test_size = 2000
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, train_size=train_size, test_size=test_size)
# print(train_data.shape, test_data.shape, train_labels.shape, test_labels.shape)
print("\nRunning KNN")
print("Training Size: " + str(train_size))
print("Testing Size: " + str(test_size))

x = []
y = []
for i in range(1, 100):
    n_neighbors = i
    print("K = " + str(n_neighbors))
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(train_data, train_labels)
    score = knn.score(test_data, test_labels)
    print("Accuracy Score = " + str(round(100 * score, 2)) + "%")
    x.append(n_neighbors)
    y.append(score)

plt.plot(x, y)
plt.show()

"""
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-",min(error_rate),"at K =",error_rate.index(min(error_rate)))
"""
