import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import KFold,cross_val_score,GridSearchCV
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../../Data/raw/Iris.csv')

data.head()

print(data.columns)

x = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = data['Species']

#other ways

#columns = data.columns.values.tolist()
#pred=columns[:4]
#target = columns[4]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

print(x_train)
print(x_test)


#Create the model and train with hyperparameters
#Criterion: entropy or gini.

tree = DecisionTreeClassifier(criterion='entropy',min_samples_split=20,random_state=99)
tree.fit(x_train,y_train)

y_pred = tree.predict(x_test) #Make the predicts

matrix = confusion_matrix(y_test,y_pred) #Make a confusion matrix
print(matrix)

print("Train Precision:",tree.score(x_train,y_train))
print("Test Precision",tree.score(x_test,y_test))

# Graphics

fig, ax = plt.subplots(figsize=(12,10))

print(f"Profundidad del arbol: {tree.get_depth()}")
print(f"Numero de nodos de decision: {tree.get_n_leaves()}")

plot = plot_tree(decision_tree=tree,
                 feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
                 class_names=['setosa', 'versicolor', 'virginica'],
                 filled=True,
                 impurity=False,
                 fontsize=11,
                 precision=2,
                 ax=ax)

# K FOLD for cross validation

tree= DecisionTreeClassifier(criterion='entropy',min_samples_split=20,random_state=99, max_depth=5)
tree.fit(x,y)

cv = KFold(n_splits=10,shuffle=True,random_state=1)
scores = cross_val_score(tree,x,y,scoring='accuracy',cv=cv.get_n_splits(x))
print("Indices de precision por cada Fold:",scores)
print("Average precision:", scores.mean())


# Another version

for i in range(1,8):
    tree = DecisionTreeClassifier(criterion='entropy', min_samples_split=20, random_state=99, max_depth=i)
    tree.fit(x, y)
    cv = KFold(n_splits=10, shuffle=True, random_state=1)
    scores = cross_val_score(tree, x, y, scoring='accuracy', cv=cv.get_n_splits(x))
    print("Depth %d score: %2f"%(i,scores.mean()))
    for j in range(0,4):
        print("  ",data.columns[j]," ", tree.feature_importances_[j])


# GridSearchCV give us the optimal parameters

param_grid = {'ccp_alpha':np.linspace(0,5,10)}

grid= GridSearchCV(
    estimator=DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=123,
    ),
    param_grid=param_grid,
    scoring='accuracy',
    cv=10,
    refit=True,
    return_train_score=True,
)

grid.fit(x_train,y_train)

final_tree = grid.best_estimator_
print(f"Tree Depth:{final_tree.get_depth()}")
print(f"Leafs: {final_tree.get_n_leaves()}")


