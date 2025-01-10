import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('../../Data/raw/Boston.csv')
data.head()

print(data.shape)

#Split data for test/train

columns = data.columns.values.tolist()
pred= columns[:4] # from 0 to 3
target = columns[4]

x = data[pred]
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

#MAX_FEATURES: cuantas caracteristicas tenemos en cuenta para tomar una decision en la division de una rama
#MAX_SAMPLES: Cada uno de los arboles utiliza 2/3 de los datos.
#OOB_SCORE: Booleano, si TRUE determinamos que debe evaluar el SCORE con los valores fuera de los 2/3

forclas = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='sqrt',max_samples=2/3, oob_score=True)
forclas.fit(x_train, y_train)

print("test",forclas.score(x_test, y_test))
print("test",forclas.score(x_train, y_train))

#graphic

for tree in forclas.estimators_:
    fig, ax = plt.subplots(figsize=(10,10))
    plot = plot_tree(
            decision_tree=tree,
            feature_names=pred,
            class_names=target,
            filled=True,
            impurity=False,
            fontsize=11,
            precision=2,
            ax=ax
        )
    plt.show()


# GRID SEARCH CV for hyperparameters

param_grid = {'n_estimators':[100,200,300],
              'max_features':[2,3],
              'max_samples':[2/3,3/4],
              'max_depth':[None,3,5,7],
              'criterion':['gini','entropy']
              }
# With cross validation

grid = GridSearchCV(
    estimator=RandomForestClassifier(n_estimators=1),
    param_grid=param_grid,
    scoring='accuracy',
    cv=10,
    refit=True,
    return_train_score=True,
)

grid.fit(x_train, y_train)

#Make a dataframe to visualize

results = pd.DataFrame(grid.cv_results_)

results.filter(regex='(param*|mean_t)') \
    .drop(columns='param') \
    .sort_values('mean_test_score',ascending=False) \
    .head()
