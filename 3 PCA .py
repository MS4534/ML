
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

dn=pd.read_csv("iris.data.csv")
dn

dn=pd.read_csv("iris.data.csv",header=None)
dn

dn.columns=['sepal_length','sepal_width','petal_length','petal_width','speciecs']
print( dn)

dn.columns=['sepal_length','sepal_width','petal_length','petal_width','speciecs']
dn.head(3)

dn1=dn.drop(columns='speciecs')

scaler=StandardScaler()
x_scaled=scaler.fit_transform(dn1)
x_scaled

pca=PCA(n_components=2)
x_pca=pca.fit_transform(x_scaled)
x_pca

dn2=pd.DataFrame(x_pca,columns=['PCA1','PCA2'])
dn2

dn2['speciecs']=dn['speciecs']
dn2['speciecs']

plt.figure(figsize=(8,6))
ns.scatterplot(x=dn2['PCA1'],y=dn2['PCA2'],hue=dn2['speciecs'],palette='dark')
plt.title(f" PCA from 4 to 2 features")
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.show()

print("Variance ratio",pca.explained_variance_ratio_)

