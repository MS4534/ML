import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("HousingData.csv")

df

x=df[['RM']].values
y=df['MEDV'].values

scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

def get_weights(X_train,query_point,tau):
    return np.exp(-np.sum((X_train - query_point) * 2,axis=1)/(2*tau*2))

def locally_weighted_regression(X_train,y_train,tau,X_test):
    y_pred=[]
    
    for x in X_test:
        weights=get_weights(X_train,x,tau)
        W = np.diag(weights)
        
        X_bias =np.c_[np.ones(X_train.shape[0]),X_train]
        theta=np.linalg.pinv(X_bias.T @ W @ X_bias)@ (X_bias.T @ W @ y_train)
        x_bias=np.array([1,x[0]])
        y_pred.append(x_bias @ theta)
    return np.array(y_pred)    

X_test=np.linspace(min(x_scaled),max(x_scaled),100).reshape(-1,1)
tau_values=[0.1,0.5,1.0]

plt.figure(figsize=(10,6))
plt.scatter(x_scaled,y,color='gray',label='Original Data')

for tau in tau_values:
    y_pred=locally_weighted_regression(x_scaled,y,tau,X_test)
    plt.plot(X_test,y_pred,label=f'LWR(t={tau})')
plt.xlabel('Scaled RM(Avg Rooms per Dweelling)')
plt.ylabel('House Price(MEDV)')
plt.title('Locally Weighted Regression on Boston Housing Dataset')
plt.legend()
plt.show()

