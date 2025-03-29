import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

# Load dataset
df = pd.read_csv("combined_features3.csv", header=None, index_col=0)

# Convert to dictionary
combined_features = {index: row.values for index, row in df.iterrows()}
feature_matrix = np.array(list(combined_features.values()))

# Standardize features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_matrix) * 5

# Apply Isolation Forest for outlier detection
forest = IsolationForest(max_samples=256, random_state=42)
outliers = forest.fit_predict(scaled_features)

valid_websites = [website for i, website in enumerate(combined_features.keys())]
valid_features = [scaled_features[i] for i in range(len(outliers)) if outliers[i] == 1]

# Apply PCA to reduce dimensionality while retaining 95% variance
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(scaled_features)

# Determine optimal number of clusters using BIC/AIC
bic_scores = []
aic_scores = []
k_values = range(2, 10)

for k in k_values:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(pca_features)
    bic_scores.append(gmm.bic(pca_features))
    aic_scores.append(gmm.aic(pca_features))

# Plot BIC and AIC scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, bic_scores, marker='o', linestyle='-', label='BIC')
plt.plot(k_values, aic_scores, marker='s', linestyle='--', label='AIC')
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('BIC and AIC for Optimal K')
plt.legend()
plt.show()

# Select the best K based on the lowest BIC score
optimal_k = k_values[np.argmin(bic_scores)]
print("Optimal K determined by BIC:", optimal_k)

# Fit Gaussian Mixture Model with optimal K
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(pca_features)

# Map websites to clusters
for i, website in enumerate(valid_websites):
    cluster_label = gmm_labels[i]
    print(f"{website} -> Cluster {cluster_label}")

# Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
reduced_features = pca_2d.fit_transform(pca_features)
plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                      c=gmm_labels, cmap='viridis', s=50, edgecolors='k')

color_bar = plt.colorbar(scatter)
color_bar.set_label('Cluster Label')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D GMM Clustering of Websites')
plt.show()

# Open websites from a specific cluster
selected_cluster = 2  # Change this to the desired cluster
cluster_websites = [valid_websites[i] for i in range(len(valid_websites)) if gmm_labels[i] == selected_cluster]

print("Websites in Cluster", selected_cluster, ":", cluster_websites)
for site in cluster_websites:  # Open only first 5 to avoid overload
    options = Options()
    driver = webdriver.Chrome(options=options)
    url = f"file:///{os.path.abspath(site)}"
    driver.get(url)

time.sleep(10000)
