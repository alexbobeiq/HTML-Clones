import pandas as pd
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from kneed import KneeLocator
from feature_vector import extract_features

print("choose tier for clustering: 1, 2, 3 or 4")

tier = int(input())

while tier not in (1, 2, 3, 4):
    print("Choose between 1, 2, 3 or 4")
    tier = input()

print("Extracting textual and image features...")
shape_text, shape_img = extract_features(tier)

# load la feature matrix
df = pd.read_csv("combined_features.csv", header=None, index_col=0)

combined_features = {index: row.values for index, row in df.iterrows()}
feature_matrix = np.array(list(combined_features.values()))

# Centrare si normare + adauga weigh 3 la features de text
text_collums = np.arange(0, shape_text)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(feature_matrix)
scaled_features[:, text_collums] = scaled_features[:, text_collums] * 3 # Adjust weight scaling


valid_websites = [website for website in enumerate(combined_features.keys())]

# Analiza componentelor principale
pca = PCA(n_components=0.95)
pca_features = pca.fit_transform(scaled_features)

# Cautare k optim prin Elbow si Silhouette
silhouette_scores = []
inertia = []
k_values = range(2, 20)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pca_features)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(pca_features, kmeans.labels_))
    print(f"K={k}, Inertia={kmeans.inertia_}, Silhouette Score={silhouette_scores[-1]}")

# Punct de elbow
knee_locator = KneeLocator(k_values, inertia, curve="convex", direction="decreasing")
optimal_k_elbow = knee_locator.elbow

# Verifica daca punctul de elbow(inflexiune) gasit este suficient de pronuntat
if optimal_k_elbow:
    elbow_idx = k_values.index(optimal_k_elbow)
    
    second_derivative = np.diff(np.diff(inertia)) # a 2 a derivata pentru toate punctele
    
    if elbow_idx > 0 and elbow_idx < len(second_derivative):
        elbow_strength = second_derivative[elbow_idx - 1] 
    else:
        elbow_strength = 0
    
    inertia_drop = (inertia[elbow_idx - 1] - inertia[elbow_idx]) / inertia[elbow_idx - 1] # cat de mult scade inertia in punctul de inflexiune

    if elbow_strength > np.percentile(second_derivative, 75) and inertia_drop > 0.4:
        optimal_k = optimal_k_elbow
        print(f"✅ Optimal K determined by Elbow Method: {optimal_k}")
    else:
        optimal_k = k_values[np.argmax(silhouette_scores)]
        print(f"🔄 Switching to Silhouette Score: Optimal K = {optimal_k}")
else:
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"🔄 No clear elbow detected. Using Silhouette Score: Optimal K = {optimal_k}") # daca Elbow point nu e pronuntat, folosim Silhuette socre 

# graficul inertiei
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker='o', linestyle='--', label="Inertia")
if optimal_k_elbow:
    plt.axvline(x=optimal_k_elbow, color='red', linestyle='--', label="Elbow Point")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.legend()
plt.show()

# graficul Silhuette score
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker='o', linestyle='--', label="Silhouette Score")
plt.axvline(x=optimal_k, color='green', linestyle='--', label="Chosen K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for Optimal K")
plt.grid(True)
plt.legend()
plt.show()

# model final
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans_labels = kmeans.fit_predict(pca_features)

for i, website in enumerate(valid_websites):
    cluster_label = kmeans_labels[i]
    print(f"{website} -> Cluster {cluster_label}")

# vizualizare clustere
pca_2d = PCA(n_components=2)
reduced_features = pca_2d.fit_transform(pca_features)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], 
                      c=kmeans_labels, cmap='viridis', s=50, edgecolors='k')

color_bar = plt.colorbar(scatter)
color_bar.set_label('Cluster Label')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('2D K-Means Clustering of Websites')
plt.show()

# output
while True:
    selected_cluster = int(input(f"Choose one one of the clusters to show: 1-{range(len(kmeans_labels))} or -1 to quit"))
    if selected_cluster == -1:
        exit()
    cluster_websites = [valid_websites[i] for i in range(len(valid_websites)) if kmeans_labels[i] == selected_cluster]

    print("Websites in Cluster", selected_cluster, ":", cluster_websites)
    