import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as ns 


df=pd.read_csv("Titanic-Dataset.csv")
df

df.isnull()

df.isnull().sum()

num_col='Fare'

mean_val=df[num_col].mean()
print('mean:',mean_val)

mode_val=df[num_col].mode()
print('mode:',mode_val)

stnad=df[num_col].std()
print('stnad:',stnad)

variance=df[num_col].var()
print('var:',variance)

range=df[num_col].max()-df[num_col].min()
print('range:',range)


med_val=df[num_col].median()
print('median',med_val)

pip install seaborn

plt.figure(figsize=(10,5))
ns.distplot(df[num_col],bins=20,kde=True,color='pink')
plt.title(f"Histogram of{num_col}")
plt.xlabel("num_col")
plt.ylabel("frequency")
plt.show
