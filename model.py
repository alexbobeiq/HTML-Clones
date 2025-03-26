import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("combined_features.csv", header=None, index_col=0)

combined_features = {index: row.values for index, row in df.iterrows()}
feature_matrix = np.array(list(combined_features.values()))
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_matrix)

forest = IsolationForest(max_samples=256)
outliers = forest.fit_predict(scaled_features)

outlier_websites = {list(combined_features.keys())[i]: outliers[i] for i in range(len(outliers)) if outliers[i] == -1}

print("Detected outlier websites:")
for website in outlier_websites:
    print(website)

valid_websites = [website for i, website in enumerate(combined_features.keys()) if outliers[i] == 1]
valid_features = [feature_matrix[i] for i in range(len(outliers)) if outliers[i] == 1]



dbscan = DBSCAN(eps=1.2, min_samples=2) 
dbscan_labels = dbscan.fit_predict(valid_features)

for i, website in enumerate(valid_websites):
    cluster_label = dbscan_labels[i]
    print(f"{website} -> Cluster {cluster_label}")


pca = PCA(n_components=3)
reduced_valid_features = pca.fit_transform(valid_features)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')


scatter = ax.scatter(reduced_valid_features[:, 0], reduced_valid_features[:, 1], reduced_valid_features[:, 2], 
                     c=dbscan_labels, cmap='viridis', s=50)

color_bar = fig.colorbar(scatter, ax=ax)
color_bar.set_label('Cluster Label')

ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
ax.set_title('3D DBSCAN Clustering of Websites (After Outlier Removal)')

plt.show()

cluster_1_websites = [valid_websites[i] for i in range(len(dbscan_labels)) if dbscan_labels[i] == 1]

for site in cluster_1_websites:
    options = Options()
    driver = webdriver.Chrome(options=options)
    url = f"file:///{os.path.abspath(site)}"
    driver.get(url)

time.sleep(10000)