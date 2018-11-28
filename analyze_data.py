#!/usr/bin/python
import csv
import json
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
import scipy.stats as st

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


#get the X and Y's for each set
process_data(df_train)
y = df_train['attack_cat']
X = df_train.drop('attack_cat',axis=1)
X_test = df_test.drop('attack_cat',axis=1)
y_test = df_test['attack_cat']


def correlation_matrix():
  plt.matshow(df_train.corr())
  plt.show()

def get_best_distribution(data):
    dist_names = ["norm","exponweib","weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)
        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best p value: "+ str(best_p))
    print("Parameters for the best fit: "+ str(params[best_dist]))

    return best_dist, best_p, params[best_dist]

protos = X['proto']
loc, scale = st.norm.fit(protos)
n = st.norm(loc=loc,scale=scale)
#plt.plot(
plt.hist(protos)#,bins=np.arange(1,130,1))
plt.show()
