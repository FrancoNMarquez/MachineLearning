import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
#Library for separate data for test and for training
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')
import io


data = pd.read_csv('../../Data/raw/Boston.csv')
data.head()

print(data.shape)

x = np.array(data[["rm"]])
y = data["medv"]

plt.scatter(x, y)
plt.xlabel('Number of rooms')
plt.ylabel('Average Price')
plt.show()

#Separate the data 20% train, 80% test
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

model = linear_model.LinearRegression() #Select the model
model.fit(X_train, y_train)  #gives the best model


print("Ecuation (Y=ax+b)")
print("y=",model.coef_," X + ",model.intercept_)

Y_pred = model.predict(X_test)

plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred,color="red")
plt.title("Simple Linear Regression")
plt.xlabel("Number of rooms")
plt.ylabel("Average Price")
plt.show()

#Test score
print("Score train model:", model.score(X_train, y_train))
print("Score test model:", model.score(X_test, y_test))

#Multiple Linear Regression

X=data[["rm","age","dis"]]
Y=data["medv"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)
modelM=linear_model.LinearRegression()
modelM.fit(X_train, y_train)

print("Ecuation (Y=a1x1+a2x2+a3x3+b):")
print("y=",modelM.coef_[0]," X1 + ",modelM.intercept_[1],"X2 + ",modelM.intercept_[2],"X3+",modelM.intercept_[3])

Y_pred = modelM.predict(X_test)
plt.scatter(X_test, y_test)
plt.plot(X_test, Y_pred,color="blue")

#Polinomial Regression

newData = pd.read_csv('../../Data/raw/auto-mpg.csv')
newData.head()

newData = newData.drop(axis=0,how="any")

newData.isNull().sum()

poly = PolynomialFeatures(degree=2)
x=poly.fit_transform(np.array(newData[["horsepower"]]))
lm = linear_model.LinearRegression()
lm.fit(x, y)
print(lm.score(x, y))


