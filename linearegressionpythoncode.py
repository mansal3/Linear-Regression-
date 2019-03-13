# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:25:42 2019

@author: manpreet.saluja
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


studentmarks=pd.read_csv("C:/Users/Manpreet.saluja/Downloads/student_scores.csv")
studentmarks
studentmarks.shape
studentmarks.head()
studentmarks.describe()

#data visualisation
plt.scatter(studentmarks.Hours,studentmarks.Scores,color='green')
plt.title('student marks with hours studied')
plt.xlabel('marks')
plt.ylabel('hours')
plt.show()

#preparing data
X=studentmarks.iloc[:,:-1].values
Y=studentmarks.iloc[:,:1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, Y_train)  
print(regressor.intercept_)  
print(regressor.coef_)  

#predict
y_pred = regressor.predict(X_test)  
df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})  
from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  
df  