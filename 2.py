import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ns

d=pd.read_csv("iris.data.csv")
d

d=pd.read_csv("iris.data.csv",header=None)
d

d.isnull().sum()

d.columns=['sepal_length','sepal_width','petal_length','petal_width','speciecs']
print( d)

d1='sepal_length'
d2='sepal_width'

plt.figure(figsize=(8,6))
ns.scatterplot(x=d[d1],y=d[d2],hue=d['speciecs'],palette='dark')
plt.title(f"Scatterplot of {d1} vs {d2}")
plt.xlabel(d1)
plt.ylabel(d2)
plt.show()

pear_cor=d[d1].corr(d[d2])
print(f"Pearson Corelation coeffienct between {d1} and {d2}:",pear_cor)

cov_matrix=d[[d1,d2]].cov()
print("\n Covarince Matrix:")
print(cov_matrix)

corr_matrix=d.drop(columns=['speciecs']).corr()
print("\n Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8,6))
ns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f")
plt.title(' Corelatin Matrix Heatmap')
plt.show()

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ns

dn=pd.read_csv("Titanic-Dataset.csv")
dn

dn1='PassengerId'
dn2='Fare'




plt.figure(figsize=(8,6))
ns.scatterplot(x=dn[dn1],y=dn[dn2],hue=dn['Embarked'],palette='dark')
plt.title(f"Scatterplot of {d1} vs {d2}")
plt.xlabel(dn1)
plt.ylabel(dn2)
plt.show()

pear_cor=dn[dn1].corr(dn[dn2])
print(f"Pearson Corelation coeffienct between {dn1} and {dn2}:",pear_cor)

cov_matrix=dn[[dn1,dn2]].cov()
print("\n Covarince Matrix:")
print(cov_matrix)

corr_matrix=dn.corr()
print("\n Correlation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8,6))
ns.heatmap(corr_matrix,annot=True,cmap='coolwarm',fmt=".2f")
plt.title(' Corelatin Matrix Heatmap')
plt.show()
