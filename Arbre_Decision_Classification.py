# -*- coding: utf-8 -*-


import pandas as pd

dataset = pd.read_csv('prediction_de_fraud.csv')

dataset.columns

X = dataset.drop('isFraud', axis=1).values

target = dataset['isFraud'].values

"""
    les colonnes X[:,1], X[:,3], X[:,6] sont des valeurs catégorielles
    on va donc les transformer avec LabelEncoder().fit_transform()
"""
from sklearn.preprocessing import LabelEncoder

Label_X = LabelEncoder()
X[:,1] = Label_X.fit_transform(X[:,1])
X[:,3] = Label_X.fit_transform(X[:,3])
X[:,6] = Label_X.fit_transform(X[:,6])

#=========================
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, target, test_size=.3, random_state=42, stratify=target)


from sklearn.tree import DecisionTreeClassifier
decTree = DecisionTreeClassifier(criterion='gini', random_state=50)

decTree.fit(x_train, y_train)
y_pred = decTree.predict(x_test)


#============ EVALUATION DU MODELE ==============
score = decTree.score(x_test, y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


from sklearn.model_selection import GridSearchCV

# On crée d'abord un dictionnaire de correspondance
grid_param = {
    'max_depth' : [1,2,3,4,5,6],
    'min_samples_leaf' : [0.02, 0.04, 0.06, 0.08]
    }
grid_object = GridSearchCV(estimator=decTree, param_grid=grid_param, scoring='accuracy', cv=10)

grid_object.fit(x_train, y_train)

grid_object.best_params_


#============================================================================#
#=================== Visualisation Arbre de Decision ========================#
#============================================================================#

# INITIALISATION DU CLASSIFIER DT
decTree_2 = DecisionTreeClassifier(criterion='gini', random_state=50, max_depth=5, min_samples_leaf=0.02)


#Adapter le Classifier aux données
decTree_2.fit(X, target)


# Extrait du nom des données predictives

X_names = dataset.drop('isFraud', axis=1)


from sklearn.tree import export_graphviz

data = export_graphviz(decTree_2, out_file=None, feature_names=X_names.columns.values, proportion=True)


import pydotplus

graph = pydotplus.graphviz.graph_from_dot_data(data)


from IPython.display import Image

Image(graph.create_png())