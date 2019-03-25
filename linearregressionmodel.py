import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns

#load data set
dataset=pd.read_csv("/Users/manpreetsaluja/Downloads/housing.csv")

#data preparation
dataset.head(3)
dataset.describe()
dataset.info()
X=dataset[['lotsize']]
Y=dataset[['price']]

#spliting the dataset
Train_X,Test_X,Train_Y,Test_Y=train_test_split(X,Y,test_size=0.3)

#data visulisation
sns.pairplot(dataset) 

plt.xlabel('plotsize')
plt.ylabel('Price')
plt.scatter(X,Y,color="red")
sns.distplot(dataset['price'])
plt.hist(X,Y)

#correlation
corr=dataset.corr()
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns)

#model building
lm=LinearRegression()
model=lm.fit(Train_X,Train_Y)
model

pred_y=lm.predict(Test_X)
plt.scatter(Test_Y,pred_y)

model.coef_
model.intercept_

from sklearn.metrics import mean_squared_error
from math import sqrt

rms = np.sqrt(mean_squared_error(Test_Y, pred_y))
print(rms)

