# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:47:20 2024

@author: Stephanie Yow
"""

import sqlite3
import pandas as pd
import numpy as np

from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import Pipeline

# custom modules
import CustomTransformers as ct
import Preprocess as pp

# establish connection to database
connect = sqlite3.connect('data/noshow.db')

# read as dataframe
df = pd.read_sql_query('select * from noshow;', connect)

# close connection
connect.close()

# remove row with all null values
df1 = df[~df['no_show'].isnull()]

# rebalance dataset

# split dataset into majority and minority classes
majority_class = df1[df1['no_show'] == 0.0]
minority_class = df1[df1['no_show'] == 1.0]

# downsample majority class
majority_downsample = resample(majority_class, 
                               replace = True, 
                               n_samples = len(minority_class), 
                               random_state = 42)
        
# form new dataset of downsampled majority and original minority classes
df_downsample = pd.concat([minority_class, majority_downsample])
        
# upweight majority class
original_weight = len(majority_class) / len(df1)
downsampling_factor = len(majority_class) / len(majority_downsample)
example_weight = original_weight * downsampling_factor
        
# add new column for example_weight
df_downsample['weight'] = [example_weight if ele == 0.0 else 0.0 for ele in df_downsample['no_show']]

# set features
features = df_downsample.drop(columns = ['no_show'])

# set target
Y = df_downsample['no_show']

# split into training and test data
x_train, x_test, y_train, y_test = train_test_split(features, Y, test_size = 0.2, random_state = 42)

# define pipeline
pipeline = Pipeline(steps = [('encode_branch', ct.EncodeBranch()),
                             ('encode_month', ct.EncodeMonth()),
                             ('nights_stayed', ct.ComputeNights()),
                             ('price_per_night', ct.ComputePricePerNight()),
                             ('encode_country', ct.EncodeCountry()),
                             ('preprocess_data', pp.preprocess),
                             ('logistic_regression', LogisticRegression(solver = 'liblinear'))
                             ])

# fit pipeline
pipeline.fit(x_train, y_train)

# predict target values
predictions = pipeline.predict(x_test)
predict_proba = pipeline.predict_proba(x_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
loss = log_loss(y_test, predict_proba)

# evaluation report
report = pd.DataFrame({'Metrics': ['Accuracy Score', 'F1 Score', 'Log Loss'], 
                       ' ': [accuracy, f1, loss]})
report.set_index('Metrics')
report.to_csv('evaluation_report.txt', sep = '\t', index = False)









