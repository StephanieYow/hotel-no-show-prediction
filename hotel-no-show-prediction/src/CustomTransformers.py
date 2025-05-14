# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 17:24:20 2024

@author: Stephanie Yow
"""

from sklearn.base import BaseEstimator
import pandas as pd

class EncodeBranch(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        X['branch'] = X['branch'].map(lambda x: 0.0 if x != 'Changi' else 1.0)
        return X

class EncodeMonth(BaseEstimator):
     
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        
        # create a dictionary of months and their numbers
        month_dictionary = {'January': 1.0,
                   'February': 2.0,
                   'March': 3.0,
                   'April': 4.0,
                   'May': 5.0,
                   'June': 6.0,
                   'July': 7.0,
                   'August': 8.0,
                   'September': 9.0,
                   'October': 10.0,
                   'November': 11.0,
                   'December': 12.0}
        
        # cleanup and add a new column of numbers corresponding to arrival_month
        X['arrival_month'] = X['arrival_month'].map(lambda x: x.capitalize())
        X['arrival_month_no'] = [month_dictionary[month] for month in X['arrival_month'] if month in month_dictionary.keys()]
        
        # add a new column of numbers corresponding to booking_month
        X['booking_month_no'] = [month_dictionary[month] for month in X['booking_month'] if month in month_dictionary.keys()]
        
        # add a new column of numbers corresponding to checkout_month
        X['checkout_month_no'] = [month_dictionary[month] for month in X['checkout_month'] if month in month_dictionary.keys()]
        
        # add a new column corresponding to months stayed
        X['months_stayed'] = X['checkout_month_no'] - X['arrival_month_no']
        X['months_stayed'] = X['months_stayed'].map(lambda x: 1.0 if x == -11.0 else x)
        
        return X   

class ComputeNights(BaseEstimator):
     
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        
        # cleanup checkout_day
        X['checkout_day'] = X['checkout_day'].map(lambda x: -1 * x if x < 0 else 1 * x)
        
        # create list of values for nights stayed
        alist = []
        
        for i in range(len(X)):
    
            if X.iloc[i, 18] == 0.0:
                duration = X.iloc[i, 6] - X.iloc[i, 4]
                alist.append(duration)
    
            elif X.iloc[i, 18] == 1.0:
                if X.iloc[i, 3] in ('January', 'March', 'May', 'July', 'August', 'October', 'December'):
                    duration = X.iloc[i, 6] - X.iloc[i, 4] + 31
                    alist.append(duration)
            
                elif X.iloc[i, 3] in ('April', 'June', 'September', 'November'):
                    duration = X.iloc[i, 6] - X.iloc[i, 4] + 30
                    alist.append(duration)
        
                else:
                    duration = X.iloc[i, 6] - X.iloc[i, 4] + 28
                    alist.append(duration)   
        
            elif X.iloc[i, 18] == 2.0:
                if X.iloc[i, 3] in ('January', 'March', 'May', 'July', 'August', 'October', 'December'):
                    duration = X.iloc[i, 6] - X.iloc[i, 4] + 31 + 30
                    alist.append(duration)
            
                elif X.iloc[i, 3] in ('April', 'June', 'September', 'November'):
                    duration = X.iloc[i, 6] - X.iloc[i, 4] + 30 + 31
                    alist.append(duration)
                
                else:
                    duration = X.iloc[i, 6] - X.iloc[i, 4] + 28 + 31
                    alist.append(duration)
        
        # add a new column corresponding to nights stayed
        X['nights_stayed'] = alist
    
        return X   

class ComputePricePerNight(BaseEstimator):
      
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):   
        
        # replace None values with '0' strings
        X['price'].fillna('0', inplace = True)

        # transform strings into floats
        # convert prices in USD to SGD using exchange rate 1 SGD = 0.745 USD
        X['price'] = X['price'].map(lambda x: float(x[5:]) if 'SGD$' in str(x) else float(x[5:]) / 0.745 if 'USD$' in str(x) else float(x))
        
        # check median price w/o 0.0 values
        X2 = X[~(X['price'] == 0.0)]
        median_price = X2['price'].describe()['50%']
        
        # impute prices of 0.0 values with median
        X['price'] = X['price'].map(lambda x: x if x != 0.0 else median_price)
        
        # add a new column corresponding to average price per night
        X['price_per_night'] = X['price'] / X['nights_stayed'] 
        
        return X

class EncodeCountry(BaseEstimator):
    
    def __init__(self):
        pass
    
    def fit(self, X, y = None):
        return self
    
    def transform(self, X):
        
        # one hot encoding
        X = pd.get_dummies(data = X, columns = ['country'], dtype = float)
        
        return X
    
    
    
    
    
    
    
    