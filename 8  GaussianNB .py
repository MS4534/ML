import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
df=pd.read_csv("iris.data.csv")
iris=load_iris()
x=iris.data
y=iris.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)   
accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy of Naive Bayes Model :{accuracy:.2f}")
