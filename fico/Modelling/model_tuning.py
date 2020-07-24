import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle

pd.options.display.max_columns = None

# read data
data = pd.read_csv('data_set_cleaned_v4.csv')

# partitioning
y = data.pop('RiskPerformance')
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=323)

# Random Forest Classifier
rfmodel = RandomForestClassifier()
rf_grid = [{'n_estimators': [x for x in range(21, 31)], 'max_features': [x for x in range(2, 11)],
            'max_depth':[x for x in range(3, 11)]}]
grid_search = GridSearchCV(rfmodel, rf_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
rf_best = grid_search.best_estimator_
rf_best_score = grid_search.best_score_
print('random forest done')

# Logistic Regression
lrmodel = LogisticRegression()
lr_grid = [{'penalty': ['l1', 'l2'], 'C': np.logspace(-4, 4, 20),
            'solver': ['liblinear']}]
grid_search = GridSearchCV(lrmodel, lr_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)
lr_best = grid_search.best_estimator_
lr_best_score = grid_search.best_score_
print('logistic regression done')

# KNN
knnmodel = neighbors.KNeighborsClassifier()
knn_grid = [{'n_neighbors': [x for x in range(1, 22, 2)]}]
grid_search = GridSearchCV(knnmodel, knn_grid, cv=10, scoring='accuracy')
grid_search.fit(X_train, y_train)
knn_best = grid_search.best_estimator_
knn_best_score = grid_search.best_score_
print('KNN done')

# SVM
SVCmodel = SVC()
SVC_grid = [{'C': [0.01, 0.1, 1], 'kernel': ['linear']},
              {'kernel': ['rbf'], 'C':[0.01, 0.1, 1, 10],
               'gamma': [1, 0.1, 0.01, 0.001]}]
grid_search = GridSearchCV(SVCmodel, SVC_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
SVC_best = grid_search.best_estimator_
SVC_best_score = grid_search.best_score_
print('SVM done')

# training whole training data
best_model = rf_best
best_model.fit(X_train, y_train)
y_test_pred = best_model.predict(X_test)
best_model_accuracy = accuracy_score(y_test, y_test_pred)
print(best_model_accuracy)

# model output
best_model.fit(X, y)
pickle.dump(best_model, open('best_model_rf.sav', 'wb'))

# see random forest feature significance
names = X.columns
print(sorted(zip(map(lambda x: round(x, 4), best_model.feature_importances_), names), reverse=True))
importances = list(best_model.feature_importances_)
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(names, importances)]
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
[print('Variable: {:40} Importance: {}'.format(*pair)) for pair in feature_importances]

