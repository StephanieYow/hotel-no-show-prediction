# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:36:00 2024

@author: Stephanie Yow
"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

preprocess = ColumnTransformer([('scale_data', StandardScaler(), ['branch', 
                                                                  'booking_month_no',
                                                                  'arrival_month_no',
                                                                  'country_Australia',
                                                                  'country_China',
                                                                  'country_India',
                                                                  'country_Indonesia',
                                                                  'country_Japan',
                                                                  'country_Malaysia',
                                                                  'country_Singapore',
                                                                  'price_per_night',
                                                                  'nights_stayed',
                                                                  'weight'])
                                               ], remainder = 'drop')
