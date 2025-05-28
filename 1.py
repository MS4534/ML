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

Q1=df[num_col].quantile(0.25)
Q3=df[num_col].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR
outlires=df[(df[num_col]<lower_bound)|(df[num_col]>upper_bound)]
print("\n outlires in the dataset")
print(outlires[num_col])

category='Survived'

cat_count=df[category].value_counts()
cat_count



plt.figure(figsize=(6,6))
plt.pie(cat_count, labels=cat_count.index, colors=["pink", "red"])
plt.title(f"pie chart of {category}")
plt.show()

plt.figure(figsize=(6,6))
plt.pie(cat_count, labels=cat_count.index,autopct="%1.1f%%" ,colors=["pink", "red"])
plt.title(f"pie chart of {category}")
plt.show()

plt.figure(figsize=(10,5))
ns.boxplot(x=df[num_col],color='yellow')
plt.title(f"boxplot {num_col}")
plt.show()
