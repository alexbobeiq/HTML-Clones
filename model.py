import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
df = pd.read_csv("combined_features.csv", header=None, index_col=0)

# Convert to dictionary
combined_features = {index: row.values for index, row in df.iterrows()}
feature_matrix = np.array(list(combined_features.values()))

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_matrix)

# Apply Isolation Forest for outlier detection
forest = IsolationForest(max_samples=256, random_state=42)
outliers = forest.fit_predict(scaled_features)
print(outliers)


valid_websites = [website for i, website in enumerate(combined_features.keys()) if outliers[i] == 1]
valid_features = [scaled_features[i] for i in range(len(outliers)) if outliers[i] == 1]
valid_websites = [website for i, website in enumerate(combined_features.keys())]
valid_features = scaled_features;
# Apply PCA to reduce dimensionality while retaining 95% variance
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(valid_features)

# Determine optimal number of clusters using Elbow Method
inertia = []
k_values = range(2, 10)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

second_derivative = np.diff(np.diff(inertia))
optimal_k = k_values[np.argmin(second_derivative) + 1]  # +1 to shift index correctly
print("Optimal K determined by Elbow Method:", optimal_k)

# Choose K and fit KMeans  # Set this based on elbow method output
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(pca_features)

# Map websites to clusters
for i, website in enumerate(valid_websites):
    cluster_label = kmeans_labels[i]
    print(f"{website} -> Cluster {cluster_label}")

# Reduce to 3D for visualization
pca_3d = PCA(n_components=3)
reduced_features = pca_3d.fit_transform(pca_features)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(reduced_features[:, 0], reduced_features[:, 1], reduced_features[:, 2], 
                     c=kmeans_labels, cmap='viridis', s=50)
color_bar = fig.colorbar(scatter, ax=ax)
color_bar.set_label('Cluster Label')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D K-Means Clustering of Websites')
plt.show()

# Open websites from a specific cluster
selected_cluster = 7  # Change this to the desired cluster
cluster_websites = [valid_websites[i] for i in range(len(valid_websites)) if kmeans_labels[i] == selected_cluster]

print("Websites in Cluster", selected_cluster, ":", cluster_websites)
for site in cluster_websites[:5]:  # Open only first 5 to avoid overload
    options = Options()
    driver = webdriver.Chrome(options=options)
    url = f"file:///{os.path.abspath(site)}"
    driver.get(url)

time.sleep(10000)
