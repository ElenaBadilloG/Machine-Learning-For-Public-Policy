import sys
import time
import numpy as np
from scipy import stats
import pandas as pd
from pandas.plotting import table
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import interactive
import seaborn as sns
import csv
import json
import requests
import urllib3
from urllib.parse import urlparse
import sklearn
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import pylab as pl
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.dummy import DummyClassifier
import itertools
import pickle



# 1. Build Classifier

def split(df, label, cols_to_drop):
    feat_cols = df.drop([label] + cols_to_drop, axis=1).columns
    X = df[feat_cols]
    y = df[label]
    return X, y

def train_test_split(df, label, test_size = 0.3,):
    X = df.drop([label], axis=1) # predictors
    Y = df[label] # label
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = 42)
    return x_train, x_test, y_train, y_test


def fit_single_classifier(mod, param, X_train, y_train):
    if mod == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=param)
    elif mod == 'DT':
        clf = DecisionTreeClassifier(criterion='gini', max_depth=param)
    elif mod == 'LR':
        clf = LogisticRegression('l2', C=param)
    elif mod == 'NB':
        clf = MultinomialNB(alpha=param)
    elif mod == 'SVM':
        clf = SVC(C=param)
        
    clf.fit(X_train, y_train)
    
    return clf
        

## 2. Evaluate Classifier

# 2.1 Very Basic:


def simple_cross_val(mod, param_lst, X_train, y_train, eval_metric):
    
    if mod == 'KNN':
        clf = KNeighborsClassifier()
        grid_values = {'n_neighbors': param_lst}
        
    elif mod == 'RF':
        clf = RandomForestClassifier()
        grid_values = {'n_estimators': param_lst}
        
    elif mod == 'LR':
        clf = LogisticRegression()
        grid_values = {'penalty': ['l1','l2'], 'C': param_lst}
        
    elif mod == 'NB':
        clf = MultinomialNB()
        grid_values = {'alpha': param_lst}
        
    elif mod == 'SVM':
        clf = SVC()
        grid_values = {'C': param_lst}
        
    elif method == 'AB':
        clf = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME")
        grid_values = {'n_estimators': param_lst}
        
    else:
        print('Error: Enter a valid model')
        return None
    
    grid = GridSearchCV(clf, grid_values, eval_metric, return_train_score = False, cv=5)
    grid.fit(X_train, y_train)
    return grid

# 2.2 Set classifier and its corresponding parameter grid

def define_clfs_params(grid_size):
    
    '''
    Define defaults for different classifiers.
    Define three types of grids:
    Test: for testing your code
    Small: small grid
    Large: Larger grid that has a lot more parameter sweeps
    
    Ref: https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    '''

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3) 
            }

    large_grid = { 
    'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], \
          'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,\
           'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],\
           'min_samples_split': [2,5,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],\
           'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],\
            'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    small_grid = { 
    'RF':{'n_estimators': [10,100], 'max_depth': [5,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,10], 'n_jobs': [-1]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.001,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1',\
                                                               'elasticnet']},
    'ET': { 'n_estimators': [10,100], 'criterion' : ['gini', 'entropy'] ,\
           'max_depth': [5,50], 'max_features': ['sqrt','log2'],\
           'min_samples_split': [2,10], 'n_jobs': [-1]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    'GB': {'n_estimators': [10,100], 'learning_rate' : [0.001,0.1,0.5],\
           'subsample' : [0.1,0.5,1.0], 'max_depth': [5,50]},
    'NB' : {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100],\
           'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],\
            'algorithm': ['auto','ball_tree','kd_tree']}
           }
    
    test_grid = { 
    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],\
          'min_samples_split': [10]},
    'LR': { 'penalty': ['l1'], 'C': [0.01]},
    'SGD': { 'loss': ['perceptron'], 'penalty': ['l2']},
    'ET': { 'n_estimators': [1], 'criterion' : ['gini'] ,'max_depth': [1], \
           'max_features': ['sqrt'],'min_samples_split': [10]},
    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
    'GB': {'n_estimators': [1], 'learning_rate' : [0.1],'subsample' : [0.5], \
           'max_depth': [1]},
    'NB' : {},
    'DT': {'criterion': ['gini'], 'max_depth': [1],'min_samples_split': [10]},
    'SVM' :{'C' :[0.01],'kernel':['linear']},
    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']}
           }
    
    if grid_size == 'large':
        return clfs, large_grid
    elif grid_size == 'small':
        return clfs, small_grid
    elif grid_size == 'test':
        return clfs, test_grid
    else:
        return 0, 0


# 2.3 K Threshold Evaluation


def joint_sort_descending(l1, l2):
    '''
    Helper Function
    Inputs:
    - l1, l2 (np.arrays)
    '''
    idx = np.argsort(l1)[::-1]
    return l1[idx], l2[idx]


def balance(df, label):
    '''
    How balanced is the label value in the whole dataset 
    '''
    return df[label].mean()

def classify_at_k(y_scores, k):
    '''
    y_true (np.array)
    y_scores (np.array)
    k (int): calssification threshold
    '''
    cutoff_index = int(len(y_scores) * k)
    predictions= [1 if x < cutoff_index else 0 for x in range(len(y_scores))]
    return predictions


def precision_at_k(y_true, y_scores, k):
    '''
    y_true (np.array)
    y_scores (np.array)
    k (int): calssification threshold
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = classify_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    return precision

def recall_at_k(y_true, y_scores, k):
    '''
    y_true (np.array)
    y_scores (np.array)
    k (int): calssification threshold
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = classify_at_k(y_scores_sorted, k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return recall

def roc_at_k(y_true, y_scores, k):
    '''
    y_true (np.array)
    y_scores (np.array)
    k (int): calssification threshold
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = classify_at_k(y_scores_sorted, k)
    roc = roc_auc_score(y_true_sorted, preds_at_k)
    return roc

def f1_at_k(y_true, y_scores, k):
    '''
    y_true (np.array)
    y_scores (np.array)
    k (int): calssification threshold
    '''
    y_scores_sorted, y_true_sorted = joint_sort_descending(np.array(y_scores), np.array(y_true))
    preds_at_k = classify_at_k(y_scores_sorted, k)
    precision = precision_score(y_true_sorted, preds_at_k)
    recall = recall_score(y_true_sorted, preds_at_k)
    return stats.hmean([precision , recall ])
    
    
def performClassification(X_train, y_train, X_test, y_test, model, param, k):
    """
    performs classification using several algorithms.
    method --> algorithm
    parameters --> list of parameters passed to the classifier (if any)

    """
   
    if model == 'RF':   
        clf = RandomForestClassifier(n_estimators=param)
        
    elif model == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=param)
        
    elif model == 'NB':
        clf = MultinomialNB(alpha= param)
    
    elif model == 'SVM':
        clf = SVC(C = param, kernel='linear', probability=True, random_state=0) 
    
    elif model == 'AB':
        clf = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME", n_estimators=param)
    
    elif model == 'GTB': 
        clf = GradientBoostingClassifier(n_estimators = param)
 
    elif model == 'LR': 
        clf = LogisticRegression(penalty='l1', C = param)
        
    elif model == 'LR2': 
        clf = LogisticRegression(penalty='l2', C = param)
    
    elif model == 'BaseL':
        clf = DummyClassifier(strategy=param,random_state=0)
        
        
    clf.fit(X_train, y_train)
    y_scores = clf.predict_proba(X_test)[:,1]
    p = precision_at_k(y_test,y_scores, k)
    r = recall_at_k(y_test, y_scores, k)
    roc = roc_at_k(y_test, y_scores, k)
    #y_pred = classify_at_k(y_scores, k)
    
    return p, r, roc
  

def clf_loop(models_to_run, clfs, grid, X, y, plott=False):
    '''
    Runs the loop using models_to_run, clfs, gridm and the data
    Ref: https://github.com/rayidghani/magicloops/blob/master/mlfunctions.py
    
    '''
    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'auc-roc','p_at_5', 'p_at_10',     'p_at_20'))
    for n in range(1, 2):
        # create training and valdation sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                
                clf.set_params(**p)
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                # you can also store the model, feature importances, and prediction scores
                # we're only storing the metrics for now
                y_pred_probs_sorted, y_test_sorted = zip(*sorted(zip(y_pred_probs, y_test),                             reverse=True))
                results_df.loc[len(results_df)] = [models_to_run[index],clf, p,
                                                       roc_auc_score(y_test, y_pred_probs), 
                                                                 precision_at_k(y_test_sorted,y_pred_probs_sorted,5.0),
                                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,10.0),
                                                                   precision_at_k(y_test_sorted,y_pred_probs_sorted,20.0)]
                                                                                                     
                    
                if plott:
                    plot_precision_recall_n(y_test,y_pred_probs,clf)

    return results_df

    
    
## 3. Temporal Splits

def time_sort(df, date_label):
    '''
    Sort a (time-labelled) dataframe by time (less to most recent obs)
    
    df (pd.DataFrame)
    date_label (str)
    '''
    df = df.sort_values(by=date_label)

def performTimeVal(X_train, y_train, folds, algorithm, param, k):
    """
    Splits X_train and y_train in a
    number of folds equal to a given number (the number of time chunks I want to initially want to split  the whole sample into). Then train on one fold and tests accuracy
    on the consecutive as follows:
    - Train on fold 1, test on 2
    - Train on fold 1-2, test on 3
    - Train on fold 1-2-3, test on 4
    ....
    Inputs:
    - X_train (pd.DataFrame)
    - y_train (pd.DataFrame)
    - folds (int)
    - algorithm (str)
    - param (float)
    - k (float)

    Returns:
    mean of test metrics.
    
    Ref: Cross Validation on Time Series (Francesco Pochetti), heavily modified.
    """
    
    # sz is the size of each fold = number of rows in X_train / folds (floored)
    sz = int(np.floor(float(X_train.shape[0]) / folds))
    print('Size of each time fold: ', sz)
    
    prec = np.zeros(folds-1)
    rec = np.zeros(folds-1)
    roc = np.zeros(folds-1)
    
 
    # loop from the first 2 folds to the total number of folds    
    for i in range(2, folds + 1):
        split = float(i-1)/i
        print('Splitting the first ' + str(i) + ' time folds at ' + str(i-1) + '/' + str(i))
        X = X_train[:(sz*i)]
        y = y_train[:(sz*i)]
        print('Size of train + test: ', X.shape) # the size of the dataframe is going to be k*i
        index = int(np.floor(X.shape[0] * split))
        
        # folds used to train the model        
        X_trainFolds = X[:index]        
        y_trainFolds = y[:index]
        
        # fold used to test the model
        X_testFold = X[(index + 1):]
        y_testFold = y[(index + 1):]
        
        prec[i-2], rec[i-2], roc[i-2] = performClassification(X_trainFolds, y_trainFolds, X_testFold, y_testFold, algorithm, param, k)

        print('Model: {}, Param: {}, K: {}, Metrics on time fold {}:, Precision = {}, Recall = {}'.format(algorithm, param, k, i , prec[i-2], rec[i-2]))
    
    # the function returns the mean of the evaluation metric on the n-1 fold    
    return prec.mean(), rec.mean(), roc.mean()

    
def basic_time_loop(X, y, clfs, thrs, folds, picklefile):
    '''
    Given a list of thresholds where 0 < t < 1, a time-sorted feature and label arrays, a set of 
    models (algorithm-parameter comnbinations), and a number of time folds to sequentially    
    estimate all the possible models and report their metrics at each threshold.
    
    Inputs:
    X (pd.DataFrame): features
    y (pd.DataFrame px1): label
    clfs (dict): algorithms and corresponding parameters to be estimated
    '''
    results_df =  pd.DataFrame(columns=('model', 'parameter', 'K', 'precision', 'recall', 'ROC'))
    for mod in clfs:
        for p in clfs[mod]:
            for k in thrs:
                r = performTimeVal(X, y, folds, mod, p, k )
                results_df.loc[len(results_df)] = [mod, p, k, r[0], r[1], r[2]]
    pickle.dump(results_df,open(picklefile, "wb"))
    return results_df


## 4. Analyze Metrics

# 4.1 Extract metrics

def get_metrics(res_df, mod, param):
    prec = 'prec_'+ mod +'_'+ str(param)
    rec = 'rec_'+ mod +'_'+ str(param)
    roc = 'roc_'+ mod +'_'+ str(param)
    prec = res_df[(res_df['model']== mod) & (res_df['parameter']==param)]['precision']
    rec = res_df[(res_df['model']==mod) & (res_df['parameter']==param)]['recall']
    roc =res_df[(res_df['model']==mod) & (res_df['parameter']==param)]['ROC']
    
    return prec, rec, roc


# 4.2 Visualize metrics


def plot_precision_recall(res_df, mod, params, xmin=0.05, xmax=1, ymin=0, ymax=1, ymin2=0, ymax2=1):
    
    #prec, rec, roc = get_metrics(res_df, mod, param)
    thrs = res_df.K.unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(params)))
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('threshold')
    ax1.set_ylabel('precision')
    ax1.tick_params(axis='y')
    ax1.set_ylim([ymin, ymax])
    ax1.set_xlim([xmin, xmax])
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('recall')
    ax2.tick_params(axis='y')
    ax2.set_ylim([ymin2, ymax2])
    for i, p in enumerate(params):
        prec, rec, roc = get_metrics(res_df, mod, p)
        ax1.plot(thrs, prec, color=colors[i], label = p)
        ax1.legend()
        ax2.plot(thrs, rec, color=colors[i])


    #ax2 = ax1.twinx()

    #color = 'tab:blue'
    #ax2.set_ylabel('recall', color=color) 
    


    fig.tight_layout()
    plt.title('Precision (<-) - Recall (->) Curves: {} = {}'.format(mod, params))
    
    plt.show()
    
def plot_model(res_df, mod, params, xmin=0, xmax=1, ymin=0, ymax=1, title='', baseline=False):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(params)))
    for i, p in enumerate(params):
        prec, rec, roc = get_metrics(res_df, mod, p)
        #bprec, brec, broc = get_metrics(res_df, 'BaseL', 'most_frequent')
        plt.plot(prec, rec, label=p, color=colors[i])
    if baseline:
        bprec, brec, broc = get_metrics(res_df, 'BaseL', 'most_frequent')
        plt.plot(bprec, brec, label='baseline', linestyle='dashed')
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.title(title)
    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.legend()

def plot_roc(res_df, mod, params, xmin=0, xmax=1, ymin=0, ymax=1, title='', baseline=False):
    
    colors = plt.cm.rainbow(np.linspace(0, 1, len(params)))
    thrs = res_df.K.unique()
    for i, p in enumerate(params):
        prec, rec, roc = get_metrics(res_df, mod, p)
        plt.plot(thrs, roc, label=p, color=colors[i])
    if baseline:
        bprec, brec, broc = get_metrics(res_df, 'BaseL', 'most_frequent')
        plt.plot(thrs, broc, label='baseline', linestyle='dashed')
    plt.ylim([ymin, ymax])
    plt.xlim([xmin, xmax])
    plt.xlabel('Threshold')
    plt.ylabel('AUC')
    plt.title(title)
    plt.legend()
    plt.show()

      