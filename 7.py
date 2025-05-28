import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df_boston=pd.read_csv("HousingData.csv")

x_boston=df_boston[['RM']].values
y_boston=df_boston['MEDV'].values
model_linear=LinearRegression()
model_linear.fit(x_boston,y_boston)
x_test=np.linspace(min(x_boston),max(x_boston),100).reshape(-1,1)
y_pred=model_linear.predict(x_test)
plt.figure(figsize=(10,5))
plt.scatter(x_boston,y_boston,color='gray',label='Original Data')
plt.plot(x_test,y_pred,color='red',label='Linear Regression')
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('House Price(MEDV)')
plt.title('Linear Regression - Boston Housing ')
plt.legend()
plt.show()

url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv"
df_mpg=pd.read_csv(url)
df_mpg=df_mpg.dropna()
x_mpg=df_mpg[['horsepower']].values
y_mpg=df_mpg['mpg'].values

scaler=StandardScaler()
x_mpg_scaled=scaler.fit_transform(x_mpg)

degrees=[2,3]
plt.figure(figsize=(10,5))
plt.scatter(x_mpg,y_mpg,color='gray',label='Original Data')

for d in degrees:
    model_poly=make_pipeline(PolynomialFeatures(d),LinearRegression())
    model_poly.fit(x_mpg_scaled,y_mpg)
    
    x_test_scaled=scaler.transform(np.linspace(min(x_mpg),max(x_mpg),100).reshape(-1,1))
    y_pred_poly=model_poly.predict(x_test_scaled)\
    
    plt.plot(np.linspace(min(x_mpg),max(x_mpg),100),y_pred_poly,label=f'Polynomial Regression(Degree{d})')
    
plt.xlabel('Horsepower')
plt.ylabel('MPG')
plt.title('Polynomail Regression -Auto MPG')
plt.legend()
plt.show()   
