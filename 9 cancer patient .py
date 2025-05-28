import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


df=pd.read_csv("/content/cancer patient data sets.csv")
df

df.isnull()

df.isnull().sum()

data=df.drop(columns=['Fatigue'])
data=data.drop(columns=['Dust Allergy'])
data.dropna(inplace=True)
X=data.drop(columns=['Patient Id', 'Level'])
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

Kmeans=KMeans(n_clusters=2,random_state=42,n_init=10)
Kmeans.fit(X_scaled)

data['cluster']=Kmeans.labels_

plt.figure(figsize=(8,6))
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=Kmeans.labels_,cmap='viridis',s=50,alpha=0.7)
plt.title('K-Means Clustering on Cancer Patient Data',fontsize=14)
plt.xlabel('Feature 1',fontsize=12)
plt.xlabel('Feature 2',fontsize=12)
plt.show()

print(f'Inertia(Within-Cluster Sum of Squuared Distances):{Kmeans.inertia_}')
