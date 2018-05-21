#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:34:50 2018
HOMEWORK 3: ML Pipeline Improvement
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
import json
import psycopg2
import sshtunnel
from sshtunnel import SSHTunnelForwarder
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
from sklearn import preprocessing, cross_validation, svm, \
metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier,  \
GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import itertools


##1. Read dataset

# 1.1 Read Files

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
    
# 1.2 SQL Queries

def sql_to_df(q, user, passw, db, localhost):
    '''
    Load an SQL table from a given database into a pandas dataframe
    '''
    conn = psycopg2.connect(host=localhost,
                               port=5432,
                               user=user,
                               password=passw,
                               database=db)

    return pd.read_sql_query(q, conn)



## 2. Explore Data

# 2.1 Dimensions and types

def overview(df, df_dups=None):
    '''
    Shows the number of rows and columns in the full dataframe and the
    duplicate rows
    dataframe, and each variable type in the full dataframe
    '''
    if df_dups is not None:
        dp = 'DUPLICATE ROWS: {}'.format(str(df_dups.shape[0]))
    else:
        dp = ''
    print('DATASET DIMENSIONS: {} rows , {} cols \n {}'.format(str(df.shape[0]), \
                                                               str(df.shape[1]),dp))
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

def to_datetime(df, var_list):
    '''
    Convert all date fields into datetime objects
    Inputs:
    - df (pd.DataFrame)
    - var_list (list of str): list of date column names
    '''
    for var in var_list:
        df[var] = pd.to_datetime(df[var],  errors='coerce')

def parse_date(td, units = 'D'):
    '''
    Convert datetime object to other time units
    
    Inputs:
    
    td (datetime obj)
    units (str): unit of interest
    '''
    if units == 'Y':
        N = 364.0
    elif units == 'M':
        N = 30.0
    else:
        N = 1
    return float(td.days)/N

# 2.1  Distributions

def cnt_dist_elem(df, var):
    '''
    Get the number of distinct unique values in a column
    '''
    return len(df[var].unique())

def percents(df, factor):
    '''
    Get what proportion of each type is there for a given factor
    in the dataset
    Inputs:
    - df (pd.DataFrame)
    - factor (str): column name of variable to group by
    '''
    for name, g in df.groupby([factor]):
        
        print("{:.3f}% {} {}".format(100.0 * len(g) / len(df), factor, name))

def cond_stats(df, vars_interest, factor):
    '''
    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of strings): list of column names
    - factor (str): column name of variable to group by
            
    Returns: table with conditional aggregate statistics
    '''
    print('SUMMARY STATISTICS FOR {} by {}:'.format(vars_interest[0].upper(), factor.upper()))
    df_subs = df[vars_interest]
    stats = df_subs.groupby(factor).agg([np.min, np.mean, np.median, np.max, np.std])
    return stats

def plot_distrib(df, var, title='', bins = 25):
    '''
    Inputs:
    - df (pd.DataFrame)
    - var (str)
    '''
    var = df[var]
    var.plot.hist(bins, alpha=0.3)
    plt.title(title)

def plot_mg_density(df, var, bw, factor=None, mxX='', minX = '', title= ''):
    '''
    Plot fitted kernel density to a variable ditribution, conditional
    on other variable.
    
    Inputs:
    - df (pd.DataFrame)
    - var (string)
    - factor (str)

    '''
    if factor:
        for n, g in df.groupby(factor):
            sns.kdeplot(g[var],shade=True, bw=bw)
            plt.xlim([mxX, minX])
            plt.legend(n)
            plt.title         
    else:
        sns.kdeplot(df[var], shade=True, color="r", bw=.15)
                
        
def plot_cond_mean(df, var, label, title='', axis_labs=['','']):
    '''
    Compute and plot the mean of a feature, grouped by
    the label's possible values.
    
    Inputs:
    - df (pd.DataFrame)
    - var , label (str)
    - title (str)
    - axis_labs (lst of str)
    '''
    plt.figure(1)
    df[[var, label]].groupby(var).mean().plot(legend=False, kind='bar')
    plt.title(title)
    plt.xlabel(axis_labs[0])
    plt.ylabel(axis_labs[1])
    
    plt.show()
        
def plot_cond_dist(df, factor, var, title, axis_labs, dens=False):
    '''
    Fit and plot a kernel density to the distribution of the label,
    grouped by the possible discretized values of a feature.
    
    Inputs:
    - df (pd.DataFrame)
    - var , label (str)
    - title (str)
    - axis_labs (lst of str)
    '''

    plt.figure(2)
    if dens:
        df.groupby(factor)[var].plot.kde()
    else:
        df.groupby(factor)[var].plot.hist(bins=25)
    plt.title(title)
    plt.xlabel(axis_labs[0])
    plt.ylabel(axis_labs[1])
    plt.show()
    

# 2.2 Time Series 

def monthly_ts(df, month_var, title):
    '''
    Plot monthly counts of observations in a dataset
    Inputs:
    - df (pd.DataFrame)
    - month_var (str)
    - title (str)
    '''
    df_gmth = pd.DataFrame({'Count' : df.groupby(['mnth_yr']).size()}).reset_index()
    ax = df_gmth.plot()
    #plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xticks([i*5.5 for i in range(15)],
           ["Jan-00", "Mar-00", "Sep-00", "Dic-00", "Jan-02", "Mar-02", \
            "Sep-02", "Dic-02","Jan-04", 
            "Mar-04", "Sep-04", "Dic-04", "Jan-06", "Mar-06", "Sep-06",\
            "Dic-06"], rotation=90)
    plt.title(title)

# 2.3 Correlations

def multi_scatter(df, vars_interest):
    '''
    Plots a scatter for each pair of selected variables, with fitted
    kernel densities
    per variable in the main diagonal.
    
    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of strings): list of column names
    '''
    df_subs = df[vars_interest]
    pd.plotting.scatter_matrix(df_subs, alpha = 0.7, figsize = (14,8), diagonal = 'kde')

def top_corr(df, vars_interest, label, heatmap = True):
    '''
    Computes pairwise correlations betweem label variable and selected
    features.
    Optionally, plots a heamap graphically depicting each pair of correlations.
    
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
    rename_cols(corr_sort, [0], ['correlation with {}'.format(label)])
    if heatmap:
        f, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                    cmap=sns.diverging_palette(220, 20, as_cmap=True))
        #plt.title('Pairwise correlation between selected variables')
        
    return corr_sort


# 2.4 Outliers

def find_outliers(df, vars_interest, thresh):
    '''
    Detect outliers for each variable in the dataset, for a given
    threshold (measured in std dev
    from the mean)

    Inputs:
    - df (pd.DataFrame)
    - vars_interest (list of str)
    - thresh (float): minimum standard deviations from the mean to be
    considered an outlier

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
                    out_lst[var] = [("MEAN:" + str(mean), "STD:" + str(std)),\
                                    ("OUTLIER", "INDEX: " ,i, "DEV: ", dev)]    
                else:
                    out_lst[var].append(( "OUTLIER", "INDEX: " , i, "DEV: ", dev))
    return  out_lst
                

## 3. Pre-Processing

# 3.1 Impute Missing Values

def na_vals(df):
    '''
    Find colums and rows with missing values. Print rows, returns list of
    columns.
    '''
    null_columns= df.columns[df.isnull().any()]
    print(df[df.isnull().any(axis=1)][null_columns])
    return null_columns
    
def impute_df(df):
    '''
    Impute missing values for the whole dataset with the median of each 
    respective column
    '''
    df = df.apply(lambda x: x.fillna(x.median(), inplace=True))
    
def simple_impute(df, var, val = 'cat'):
    '''
    Fill in missing observations in a given column with a specified value
    - val: {'med', 'mean', 0, 'cat'}
    '''
    
    if val == 'med':
        r = df[var].median()
        df[var] = df[var].apply(lambda x: r if pd.isnull(x) else x)
    elif val == 'mean':
        r = df[var].mean()
        df[var] = df[var].apply(lambda x: r if pd.isnull(x) else x)
    elif val =='cat':
        r = df[var].mode().iloc[0]
        df[var] = df[var].apply(lambda x: r if pd.isnull(x) else x)
    else:
        df[var] = df[var].apply(lambda x: 0 if pd.isnull(x) else x)
    
def standarize(v,mn,st):
    return (v - mn) / st

# 4. Generate Features/Predictors

def daytime_vals(df, date_label, tvals):
    '''
    Extract time features from a datetime variable.
    
    - df (pd.DataFrame)
    - date_label (str)
    - tvals (lst) {d, m, y, h, wd}
    '''
    df[date_label] = pd.to_datetime(df[date_label]) 
    if 'd' in tvals:
        df[ date_label +'_day'] = df[date_label].apply(lambda t: t.day)
    if 'm' in tvals:
        df[date_label +'_month'] = df[date_label].apply(lambda t: t.month)
    if 'y' in tvals:
        df[date_label +'_year'] = df[date_label].apply(lambda t: t.year)
    if 'h' in tvals:
        df[date_label +'_hour'] = df[date_label].apply(lambda t: t.hour)
    if 'wd' in tvals:
        df[date_label +'_day_of_week'] = df[date_label].dt.weekday_name
        
def hourday(x):
    '''
    Cathegorize an hour of day into morning, noon, night
    '''
    if 12 >= x >= 6:
        h = 0 # morning
    elif 13>= x >= 18:
        h = 1 # noon
    else:
        h = 2 # night
    return h

def n_most_freq(df, factor, idv, n):
    most_freq = df.groupby(factor).count()[idv] / len(df)
    most_freq = most_freq.sort_values(ascending=False).head(n)
    return most_freq
        
# 4.1 Discretization and Cathegorization

def cap_values(x, cap_min, cap_max):
    '''
    Cap a value with given min, max values.
    '''
    if x > cap_max or x < cap_min:
        return cap
    else:
        return x
    
def others_dummy(x,lst):
    '''
    For cathegorical features with many possible (but few relevant)
    values, cathegorize the important ones and the rest as 'OTHERS'
    Inputs:
    - lst (list of relevant values)
    Reurns
    - string
    '''
    for st in lst:
        if x == st:
            return x
        else:
            return 'OTHER'
        
def categorize(df, vars_to_cat, bins, vars_to_cap=[]):
    '''
    Build evenly spaced buckets for selected continous variables in a dataframe,
    cathegorize all selected variables with dummies, and add the new categorical
    variables to a dataframe.
    '''
    lst , lst_d = [], []
    for i, var in enumerate(vars_to_cat):
        name = var + '_cat'
        lst.append(name)
        name_d = var + '_dum'
        lst_d.append(name_d)
        
        if var in vars_to_cap:
            df[var] = df[var].apply(lambda x: cap_values(x, df[var].quantile(.05), df[var].quantile(.95)))
            col = pd.cut(df[var], bins[i])
            df[name] = col
            df[name_d] = df[name].cat.codes
        else:
            col = pd.cut(df[var], bins[i])
            df[name] = col
            df[name_d] = df[name].cat.codes
            
def dummify(df,var_lst):
    '''
    Assign discrete values  to cathegorical data.
    Ex: if d[A] = 'red' => d[B] = 2
    '''
    for var in var_lst:
        name_d = var + '_dum'
        df[name_d] = df[var].astype('category').cat.codes

def binarize(df,var):
    '''
    Construct binary valued-columns for cathegorical data.
    Ex: if d[A] = 'red' => d['red] = 1
    '''
    for val in df[var].unique()[1:]:
        name_d = str(val) + '_dum'
        df[name_d] = np.where(df[var]== val, 1, 0)

## 4.2 Augmenting Datasets

def parse_json_file(url):
    resp = requests.get(url=url)
    data = resp.json() # Check the JSON Response Content
    return data

## 4.3 Visualizing Features
    
def count_by(df, var, title, axis_labs):
    '''
    Compute and plot the number of ocurrences in the data, grouped by
    the possible values of a variable.
    
    df (pd.Dataframe)
    var (str)
    title (str)
    axis_labs (lst): x and y axis labels (str)
    
    '''
    pd.value_counts(df[var]).plot(kind='bar')
    plt.title(title)
    plt.xlabel(axis_labs[0])
    plt.ylabel(axis_labs[1])

    
def graph_by_fact(df, factor, var, axis_labs, title):
    '''
    Count and plot number of ocurrences in the data as per some variable,
    grouped by the possible values of a categorical feature.
    '''
    colors = plt.cm.GnBu(np.linspace(0, 1, len(df[factor].unique())))
    df_gmth = pd.DataFrame({'Count' : df.groupby([var, factor]).size()}).reset_index()
    #df_gmth = df_gmth[df_gmth.Count >=0]
    df_gmth_piv = df_gmth.pivot(index=var, columns=factor, values='Count')
    df_gmth_piv.plot(kind='bar', stacked=True, color = colors)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.xlabel(axis_labs[0])
    plt.ylabel(axis_labs[1])
    plt.title(title)

def plot_percents(df, var, label, T, ymin, ymax, title='', size=(8,5), \
                  sort_val= 'Percent'):
    '''
    Plots the proportion of ocurrences of certain value conditional
    to some value of the label.
    
    Inputs:
    
    df (pd.DatFrame)
    factor (str)
    lat_lab (str): label for latitude 
    lon_lab (str): label for longitude
    
    ''' 
    df_p = pd.DataFrame({'Percent' : df.groupby([label, var]).size()\
                      / df.groupby([var]).size() }).reset_index().sort_values(sort_val)
    df_p = df_p[df_p[label]== T].set_index(var)
    df_p['Percent'].plot(kind='bar', figsize = size)
    plt.title(title)
    #plt.rcParams["figure.figsize"] = size
    plt.ylim([ymin, ymax])
    return df_p


    
def spatial_scatter(df,factor, lat_lab, lon_lab):
    '''
    df (pd.DatFrame)
    factor (str)
    lat_lab (str): label for latitude 
    lon_lab (str): label for longitude
    
    '''  
    groups = df.groupby(factor)
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.margins(0.05)
    for name, group in groups:
        if name == df[factor].unique()[0]:
            ax.plot(group[lat_lab], group[lon_lab], marker='o', linestyle='', ms=2, \
                    label=name, alpha = 0.4)
        elif name == df[factor].unique()[1]:
            ax.plot(group[lat_lab], group[lon_lab], marker='x', linestyle='', ms=5, \
            label=name, alpha = 0.8)
        else:
            ax.plot(group[lat_lab], group[lon_lab], marker='o', linestyle='', ms=2, \
            label=name, alpha = 0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    ax.legend(fontsize=16)
    ax.set_xlabel("Latitude")
    ax.set_ylabel("Longitude")
    plt.legend(bbox_to_anchor=(1, 1), loc=0, borderaxespad=1.)
    plt.title("Spatial Distribution by {}".format(factor.upper()), fontsize=16)
    
def plot_importances(df, features, label, n=10, title=''):
    '''
    Build a random forest classifier to
    compute the relative importance of selected features in
    predicting the label.
    
    Inputs:
    - df (pd.DataFrame)
    - features (lst of str)
    - label (str)
    - n (int): top n features, opt
    - title (str)
    '''
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(df[features], df[label])
    importances = clf.feature_importances_
    np_features = np.array(features)
    sorted_idx = np.argsort(importances)[len(np_features)-n:]
    padding = np.arange(len(sorted_idx)) + 0.5
    pl.barh(padding, importances[sorted_idx], align='center')
    pl.yticks(padding, np_features[sorted_idx])
    pl.xlabel("Relative Importance")
    pl.title(title)
    pl.show()