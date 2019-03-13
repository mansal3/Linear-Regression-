# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 15:29:05 2019

@author: manpreet.saluja
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

dataset=pd.read_csv("C:/Users/Manpreet.saluja/Downloads/50_Startups.csv")
dataset
dataset.shape
dataset.describe()

#preparing data set
X=dataset[['R&D Spend','Administration','Marketing Spend']]
Y=dataset[['Profit']]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)

reg.coef_
coeff_df = pd.DataFrame(reg.coef_)  

#predicting
y_pred=reg.predict(X_test)
df = pd.DataFrame( Y_test, y_pred)  

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  