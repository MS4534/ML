import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("Breast_Cancer.csv") 
df.drop(columns=['id'], inplace=True) 
df.drop(columns=['Unnamed: 32'], inplace=True)
df.dropna(inplace=True)
X = df.drop(columns=['diagnosis'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.7)
plt.title('K-Means Clustering on Wisconsin Breast Cancer Data', fontsize=14)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.show()
