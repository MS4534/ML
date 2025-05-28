from sklearn.naive_bayes import GaussianNB
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=40)

model=GaussianNB()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)                                                                                                                                                             accuracy=accuracy_score(y_test,y_pred)
print(f"Accuracy of Naive Bayes Model :{accuracy:.2f}")
