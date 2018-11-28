#!/usr/bin/python
import csv
import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

def getFile(num):
  return "unsw/UNSW-NB15_{filenum}.csv".format(filenum=num)

def categories_to_numbers(values):
  values.sort()
  replacements = {}
  for i in range(len(values)):
    replacements[values[i]] = i
  return replacements

df_train = pd.read_csv('UNSW_NB15_training-set.csv')
df_test = pd.read_csv('UNSW_NB15_testing-set.csv')


#replaces the 
def process_data(df):
  services = df.service.unique()
  service_replacements = categories_to_numbers(services)
  df['service'].replace(service_replacements,inplace=True)
  
  #replace protocols with categories
  protos = df.proto.unique()
  protos_replacements = categories_to_numbers(protos)
  df['proto'].replace(protos_replacements,inplace=True)
  
  attack_cats = df.attack_cat.unique()
  attack_cats_replacements = categories_to_numbers(attack_cats)
  df['attack_cat'].replace(attack_cats_replacements,inplace=True)
  
  states = df.state.unique()
  states_replacements = categories_to_numbers(states)
  df['state'].replace(states_replacements,inplace=True)


#The test and training data
#clean the training and test data
process_data(df_train)
process_data(df_test)
#get the X and Y's for each set
y = df_train['attack_cat']
X = df_train.drop('attack_cat',axis=1)
X_test = df_test.drop('attack_cat',axis=1)
y_test = df_test['attack_cat']


def trainDecisionTree():
  dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
  dt.fit(X, y)
  return dt
 
def trainRandomForest():
  rf = RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)
  rf.fit(X,y)
  return rf

dummy = pd.get_dummies(df_train['service'])
print dummy.head()
'''
rf = trainDecisionTree()
y_predict = rf.predict(X_test)
print accuracy_score(y_test,y_predict)
'''








