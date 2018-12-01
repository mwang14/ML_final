import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.grid_search import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold
from sklearn.svm import SVC
# from sklearn.neural_network import
import pandas as pd
import numpy as np
import pprint as pp


def run_kfolds(clf):
    kf = KFold(891, n_folds=10)
    outcomes = []
    fold = 0
    for train_index, test_index in kf:
        fold += 1
        X_train, X_test = X_all.values[train_index], X_all.values[test_index]
        y_train, y_test = y_all.values[train_index], y_all.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print('Fold {0} accuracy: {1}'.format(fold, accuracy))
    mean_outcome = np.mean(outcomes)
    print('Mean accuracy: ', mean_outcome)

df_train = pd.read_csv('UNSW_NB15_training-set.csv')
df_test = pd.read_csv('UNSW_NB15_testing-set.csv')

X_all, y_all = preprocessing.gen_X_and_y(df_train)
X_test_all, y_test_all = preprocessing.gen_X_and_y(df_test)

num_test = .2

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test)
# print('After splitting training and testing sets')
scores = []

clf_list = [LinearSVC(), DecisionTreeClassifier(), GaussianNB(), RandomForestClassifier()]

# clf_list = [DecisionTreeClassifier()]
print('Classifiers list: ', clf_list)

parameters = {'n_estimators': [4, 6, 9],
              'max_features': ['log2', 'sqrt', 'auto'],
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10],
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1, 5, 8]
              }

# run_kfolds(clf)
# acc_scorer = make_scorer(accuracy_score)
# grid_obj = GridSearchCV(clf_list, parameters, scoring=acc_scorer)
# grid_obj = grid_obj.fit(X_train, y_train)
# clf = grid_obj.best_estimator_
accuracy = {}
for clf in clf_list:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test_all)
    score = accuracy_score(y_test_all, y_pred)
    accuracy[clf] = score
    print(confusion_matrix(y_test_all, y_pred))

pp.pprint(accuracy)
# print('accuracy score: ', accuracy_score(y_test_all, y_pred))


