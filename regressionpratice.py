# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 15:51:13 2019

@author: manpreet.saluja
"""

#linear Regression
import seaborn as sb
from matplotlib import pyplot as plt

dataset=sb.load_dataset('tips')

#datavisualisation
sb.regplot(x='total_bill',y='tip',data=dataset)

#SUMMARY OF DATASet
dataset.describe()
#no of rows and columns
dataset.shape

#data preparing
X=dataset[['total_bill','tip']]
Y=dataset[['size']]

from sklearn.model_selection import train_test_split
Train_X,Test_X,Train_Y,Test_Y=train_test_split(X,Y,test_size=0.4)

#check shape
Train_X.shape
Train_Y.shape
Test_X.shape
Test_Y.shape

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(Train_X,Train_Y)

#coefficients
lm.coef_

y_pred=lm.predict(Test_X)
df=pd.DataFrame(Test_Y,y_pred)

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Test_Y, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(Test_Y, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Test_Y, y_pred)))  