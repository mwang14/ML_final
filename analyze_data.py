#!/usr/bin/python
import csv
import json
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.cross_validation import train_test_split
# from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest,chi2
import scipy.stats as st
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

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

def getMeanAndMedian(df):
  result = {}
  for column in df:
    mean_and_median = {}
    mean=df[column].mean() 
    median=df[column].median()
    mean_and_median['mean'] = mean
    mean_and_median['median'] = median
    result[column] = mean_and_median
  return result

#print df_train.dtypes
def get_correlations(df):
  df.corr().to_csv('correlations.csv')

def correlation_matrix(df):
    from matplotlib import cm as cm

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet', 100)
    cax = ax1.imshow(df.corr(), interpolation="nearest", cmap=cmap)
    ax1.grid(True)
    plt.title('Network Intrusion Feature Correlation')
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    fig.colorbar(cax)
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

def isNormal():
  result = {}
  for column in X:
    protos = X[column]
    loc, scale = st.norm.fit(protos)
    n = st.norm(loc=loc,scale=scale)
    x = np.arange(1,len(protos.unique()),1)
    #plt.hist(protos,bins=x)
    #plt.plot(x,100000*n.pdf(x))
    alpha = .01
    D,p = st.normaltest(protos) 
    result[column] = p
  print result
  return result
    
def tryPCA():
  colors = ['navy', 'turquoise', 'darkorange']
  lw = 2
  pca = PCA(n_components=3) 
  pca.fit(X)
  X_pca = pca.transform(X)
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.scatter(X_pca[:,0],X_pca[:,1],X_pca[:,2])
  print pca.explained_variance_ratio_
  plt.show()

def tryLDA():
  lda = LinearDiscriminantAnalysis(n_components=3)
  lda.fit(X,y)
  X_lda = lda.transform(X)
  fig = plt.figure()
  ax = Axes3D(fig)
  colors = [(0,0,0),(1,1,1),(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(.5,.5,.5),(.75,0,1)]
  ax.scatter(X_lda[:,0],X_lda[:,1],X_lda[:,2])
  plt.show()

  
#replaces the 
def process_data(df):
  services = df.service.unique()
  service_replacements = categories_to_numbers(services)
  df['service'].replace(service_replacements,inplace=True)
  
  #replace protocols with categories
  protos = df.proto.unique()
  protos_replacements = categories_to_numbers(protos)
  df['proto'].replace(protos_replacements,inplace=True)

  state = df.state.unique()
  protos_replacements = categories_to_numbers(state)
  df['state'].replace(protos_replacements,inplace=True)
  
  attack_cats = df.attack_cat.unique()
  attack_cats_replacements = categories_to_numbers(attack_cats)
  df['attack_cat'].replace(attack_cats_replacements,inplace=True)
  df['service'] = df['service'].astype('category')
  df['proto'] = df['proto'].astype('category')
  df['attack_cat'] = df['attack_cat'].astype('category')
  df['state'] = df['state'].astype('category')

  continuous_columns = list(df.columns.values)
  continuous_columns.remove('service')
  continuous_columns.remove('proto')
  continuous_columns.remove('attack_cat')
  continuous_columns.remove('state')
  continuous_columns.remove('label')
  for column in continuous_columns:
    df[column] = pd.cut(df[column],100,labels=False)


process_data(df_train)
process_data(df_test)




df_train = df_train.drop('id',axis=1)
df_test = df_test.drop('id',axis=1)
df_train.to_csv('train_cleaned.csv')
df_test.to_csv('test_cleaned.csv')
#y = df_train['attack_cat']
#y2 = df_train['label']
#X = df_train.drop(['attack_cat','label'],axis=1)
#
#X_test = df_test.drop(['attack_cat','label'],axis=1)
#
#y_test = df_test['attack_cat']
#y_test2 = df_test['label']
#
#X = pd.DataFrame(preprocessing.scale(X))
#X_test = pd.DataFrame(preprocessing.scale(X_test))
#

