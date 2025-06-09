import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import io

#SVM

iris = pd.read_csv('../../Data/raw/Iris.csv')

x = iris[["Sepal.Length","Sepal.Width"]]
y = iris[["Species"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Try with Linear Kernel
svc = svm.SVC(kernel='linear', C=100, gamma=1000) #Hyperparameters Kernel , C= , Gamma= , high values can make overfit
svc.fit(x_train, y_train)
print("Precision train model:", svc.score(x_train, y_train))
print("Precision test model:", svc.score(x_train, y_train))

# Try with Gaussian Kernel
svc = svm.SVC(kernel='rbf', C=100, gamma=1000) #Hyperparameters Kernel , C= , Gamma=
svc.fit(x_train, y_train)
print("Precision train model:", svc.score(x_train, y_train))
print("Precision test model:", svc.score(x_train, y_train))


#Lets find better C y Gamma

svcG = svm.SVC(kernel='rbf')
svcG.fit(x_train, y_train)

param = {
    "C":[0.001,1,10,100],
    "gamma":[0.1,1,5]
}
grid = GridSearchCV(svc, param)

# Train this new model
grid.fit(x_train, y_train)

print (grid.best_params_) # These are the best hyperparameters

# confusion matrix

y_pred = grid.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Classification problem using MNIST DataSet ( Digits handwritten)

mnist = fetch_openml("mnist_784",version=1,cache=True,as_frame=False)
print(mnist.keys())

print(mnist.shape())
# reconfig 784 positions into an 28x28 matrix
image = mnist.data[0].reshape(28,28)

plt.imshow(image, cmap='binary')

print(mnist.target[0])


x_train = mnist['data'][:60000]
y_train = mnist['target'][:60000]

x_test = mnist['data'][60000:]
y_test = mnist['target'][60000:]

lin_clf = svm.SVC(kernel='rbf',random_state=43)
lin_clf.fit(x_train, y_train)

print("Prediction train model:",lin_clf.score(x_train, y_train))
print("Prediction test model:",lin_clf.score(x_test, y_test))

scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.astype(np.float32))
x_test_scaled = scaler.transform(x_test.astype(np.float32))

lin_clfG = svm.SVC(kernel='rbf')
lin_clfG.fit(x_train_scaled, y_train)



