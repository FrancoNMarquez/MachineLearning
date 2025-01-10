from scipy.special import y_pred
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

import pandas as pd
import numpy as np

data = datasets.load_breast_cancer()
print(data.keys())

df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()

pd.crosstab(data.target, columns="count")

X=data.data
y=data.target

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# We need to scale the data, values between 0 and 1
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

logit_model=LogisticRegression()
logit_model.fit(x_train, y_train)

print(logit_model.score(x_train, y_train))
print(logit_model.score(x_test, y_test))

y_pred=logit_model.predict(x_test)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)


#NonBinary output
#Need to find the csv used in the course ( class 7 )