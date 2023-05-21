import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# Read the data from the CSV file
data = pd.read_csv('sample_data.csv')

# Select all columns except 'Time' and 'Shard Split Needed?'
features = data.drop(['Time', 'Shard Split Needed?'], axis=1)

# Normalize the selected features
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(features)

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(normalized_features)

# Get the cluster labels for each data point
cluster_labels = kmeans.labels_

# Add the cluster labels to the original dataset
data['Cluster'] = cluster_labels

print(data)

# Create a scatter plot of the clusters
plt.figure(figsize=(12, 8))
plt.scatter(data.index, data['Shard Size (GB)'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Data Point Index')
plt.ylabel('Shard Size (GB)')
plt.title('K-means Clustering')
plt.colorbar()
plt.show()