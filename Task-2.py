import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

# Step 1: Loading the dataset
data = pd.read_csv('Mall_Customers.csv')

# Step 2: Preparing the dat
# Replacing the original gender column and adding a new gender column in its place with numerical values 
encoder = OneHotEncoder(drop='first')
encoded_gender = encoder.fit_transform(data[['Gender']]).toarray()
encoded_gender_df = pd.DataFrame(encoded_gender, columns=encoder.get_feature_names_out(['Gender']))

# Concatenating the encoded gender columns with the original dataframe
data = pd.concat([data, encoded_gender_df], axis=1)
# Droping the original 'Gender' column
data.drop('Gender', axis=1, inplace=True)

# Selecting relevant features for clustering
features = data[['Annual Income (k$)', 'Spending Score (1-100)'] + list(encoded_gender_df.columns)]  # Adjust according to your dataset

# Step 3: Determining the optimal number of clusters using the elbow method
sse = []
k_range = range(1, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    sse.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(k_range, sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal k')
plt.show()

# Step 4: Performing K-means clustering with the chosen number of clusters
optimal_k = 5 
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)

# Step 5: Analyzing the results
print(data.groupby('Cluster').mean())

# Visualizing the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()