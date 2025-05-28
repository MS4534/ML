import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,f1_score

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

data=load_iris()
data

x=data.data
y=data.target

print(x)

print(y)

std_scalar=StandardScaler()
x_scaled=std_scalar.fit_transform(x)
x_scaled

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

def evaluate_knn(k_values,weighted='False'):
    results=[]
    for k in k_values:
        if weighted:
            model=KNeighborsClassifier(n_neighbors=k,weights='distance')
        else:
            model=KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)
        acc=accuracy_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred,average='weighted')
        results.append((k,acc,f1))
        print(f"k=(k)|Accuracy:{acc:4f}|F1-score{f1:.4f}")
    return results 

k_values=[1,3,5]
print("\n Regular KNN Results")
regular_knn_result=evaluate_knn(k_values,weighted='uniform')
print("\n weighted knn results")
weighted_knn_result=evaluate_knn(k_values,weighted=True)

