#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:34:50 2018
HOMEWORK 2: ML Pipeline
@author: elenabg
"""
import sys
import time
import numpy as np
import pandas as pd
from pandas.plotting import table
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import interactive
import seaborn as sns
import csv
import sklearn
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pylab as pl
from sklearn.metrics import accuracy_score as accuracy
from sklearn.preprocessing import StandardScaler

#import graphviz 

# #1. Read dataset

def load_data(path, filename, format_ = 'csv', dups = False):
    '''
    Reads data from an external source into pandas.
    
    Inputs:
    - path (str) 
    - filename (str)
    - format_ (str): csv (default), json, stata, excel
    - dups (bool): False (default)
    
    Returns:
    - pandas.DataFrame with full dataset
    - pandas.DataFrame with duplicate rows, if dups is set to True
    '''
    if format_ == 'csv':
        df_all = pd.read_csv(path + filename)
    elif format_ == 'json':
        df_all = pd.read_json(path + filename)
    elif format_ == 'stata':
        df_all = pd.read_stata(path + filename, convert_categoricals = False)
    elif format_ == 'excel':
        df_all = pd.read_excel(path + filename)

    if dups:
        df_all['Dup'] = df_all.duplicated(subset= df_all.columns, keep = False)
        df_dups = df_all[df_all.Dup == True]
        df_all = df_all.drop(labels=['Dup'], axis=1)
        return df_all, df_dups
    else:
        return df_all


## 2. Explore Data

# 2.1 Dimensions and types

def overview(df, df_dups):
    '''
    Shows the number of rows and columns in the full dataframe and the duplicate rows
    dataframe, and each variable type in the full dataframe
    '''
    print('DATASET DIMENSIONS: ' + str(df.shape[0]) + ' rows' ', ' + str(df.shape[1]) + \
          ' columns'  + '\n' +'DUPLICATE ROWS:' + str(df_dups.shape[0]))
    return df.dtypes

def rename_cols(df, col_list, new_names):
    '''
    Inputs:
    - df (pd.DataFrame)
    - col_list (list of int): list of indexes
    - new_names (list of str): list of new columns names
    
    Returns: dataframe with renamed columns
    '''
    cols = [df.columns[i] for i in col_list]
    new_cols = dict(zip(cols, new_names))
    df.rename(columns = new_cols, inplace = True)

# 2.1  Distributions

def cond_stats(df, vars_interest, factor):
    '''
    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of strings): list of column names
    - factor (str): column name of variable to group by
            
    Returns: table with conditional aggregate statistics
    '''
    
    df_subs = df[vars_interest]
    stats = df_subs.groupby(factor).agg([np.min, np.mean, np.median, np.max])
    return stats

def plot_distrib(df, var, bins = 25, cap=None):
    '''
    Plot histogram and fitted kernel density to a variable ditribution.
    
    Inputs:
    - var (np.array)
    '''
    var = df.apply(lambda x: x.fillna(x.median()))[var]
    var.plot.hist(bins, alpha=0.3)
    plt.title(var.name)

def plot_mg_density(df, var, factor):
    '''
    Plot fitted kernel density to a variable ditribution, conditional on other variable.
    
    Inputs:
    - df (pd.DataFrame)
    - var (string)
    - factor (str)

    '''
    if factor:
        df.groupby(factor)[var].plot.kde()
        plt.title(str(var).upper() +'by' + str(factor).upper())
        plt.legend()
    else:
        df[var].plot.kde()
        plt.title(str(var).upper())

# 2.2 Correlations

def multi_scatter(df, vars_interest):
    '''
    Plots a scatter for each pair of selected variables, with fitted kernel densities
    per variable in the main diagonal.
    
    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of strings): list of column names
    '''
    df_subs = df[vars_interest]
    pd.plotting.scatter_matrix(df_subs, alpha = 0.7, figsize = (14,8), diagonal = 'kde')

def top_corr(df, vars_interest, label, heatmap = True):
    '''
    Computes pairwise correlation betweem label variable and selected
    features.
    Optionally, plots a heamap for each pair of correlations.
    
    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of str): list of column names
    - label (str)

    Returns:
    - pd. Dataframe with sorted correlations
    '''
    df_subs = df[vars_interest]
    corr = df_subs.corr()
    corr_sort = df_subs.corr()[label].sort_values(ascending = False).to_frame()
    rename_cols(corr_sort, [0], ['correlation w/ delinquency'])
    if heatmap:
        f, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 20, as_cmap=True),
        ax=ax)
    return corr_sort


# 2.3 Outliers

def find_outliers(df, vars_interest, thresh):
    '''
    Detect outliers for each variable in the dataset, for a given threshold (measured in std dev
    from the mean)

    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of str)
    - thresh (float): minimum standard deviations from the mean to be considered an outlier

    Returns:
    - dict with variable as key and a list containg each outlier index
      and deviation magnitude as val.
    '''
    out_lst = {}
    df_subs = df[vars_interest]
    for var in df_subs.columns:
        mean = df[var].mean()
        std = df[var].std()
        
        for i, row in df_subs.iterrows():
            dev = abs((df_subs[var].loc[i] - mean) / std )
            if dev > thresh:
                if var not in out_lst:
                    out_lst[var] = [("MEAN:" + str(mean), "STD:" + str(std)), ("OUTLIER", "INDEX: " ,i, "DEV: ", dev)]    
                else:
                    out_lst[var].append(( "OUTLIER", "INDEX: " , i, "DEV: ", dev))
    return  out_lst
                

# 3. Pre-Processing

# 3.1 Impute Missing Values

def standarize(var):
    return (var - var.mean()) / var.std()

def impute(df, method = 'simple'):
    '''
    Fill in missing observations with the sample median
    '''
    if method =='simple':
        df = df.apply(lambda x: x.fillna(x.median()))
    return df

# 4. Generate Features/Predictors
    
# 4.1 Discretization and Cathegorization

def cap_values(x, cap):
    '''
    Cap a value with a given max value.
    '''
    if x > cap:
        return cap
    else:
        return x   

def categorize(df, vars_to_cat, bins):
    '''
    Build evenly spaced buckets for selected continous variables in a dataframe,
    cathegorize all selected variables  with dummies, and add the new categorical
    variables to a dataframe.
    '''
    lst , lst_d = [], []
    for i, var in enumerate(vars_to_cat):
        name = var + '_cat'
        lst.append(name)
        name_d = var + '_dum'
        lst_d.append(name_d)
        
        if var == 'month_inc' or var == 'rev_util' or var == 'debt_ratio':
            df[var] = df[var].apply(lambda x: cap_values(x, df[var].quantile(.95)))
            col = pd.cut(df[var], bins[i])
            df[name] = col
            df[name_d] = df[name].cat.codes
        else:
            col = pd.cut(df[var], bins[i])
            df[name] = col
            df[name_d] = df[name].cat.codes
            
    df_dums = df[['ser_DLq2'] + lst_d]
    return df_dums


# 4.2 Visualize Features
    
def hist(df, var):
    pd.value_counts(df[var]).plot(kind='bar')
    
def plot_cond_mean(df, var, label):
    
    plt.figure(1)
    df[[var, label]].groupby(var).mean().plot(kind='bar')
    plt.title('Average ' + label + ' by level of ' + var + '\n' + '(intervals - to +)')
    plt.show()
        
def plot_cond_dist(df, var, label):

    plt.figure(2)
    df.groupby(var)[label].plot.kde()
    plt.title(var +  ' by ' + label + ' status'+ '\n' +' (intervals - to +)')
    plt.legend(fontsize=9)
    plt.show()
    
def plot_importances(df, features, label):
    clf = RandomForestClassifier()
    clf.fit(df[features], df[label])
    importances = clf.feature_importances_
    sorted_idx = np.argsort(importances)
    padding = np.arange(len(features)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title("Variable Importance")
    pl.show()

# 5. Build Classifier

def split_dataset(df, label, test_size = 0.3,):
    X = df.drop([label], axis=1) # predictors
    Y = df[label] # label
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = test_size, random_state = 42)
    return x_train, x_test, y_train, y_test

def fit_classifier(model, x_train, y_train):
    if model == 'KNN':
        clf = KNeighborsClassifier(n_neighbors=10)
    elif model == 'DT':
        clf = DecisionTreeClassifier(criterion='gini', max_depth=3)
    clf.fit(x_train, y_train)
    return clf
        

# 6. Evaluate Classifier

def get_acc(clf, x_test, y_test):
    y_pred = clf.predict(x_test)
    tot = len(y_test)
    good = [i for i in range(len(y_pred)) if y_pred[i] == y_test.iloc[i]]
    acc = len(good) / tot
    return acc    
