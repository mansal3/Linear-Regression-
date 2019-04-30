#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 07:43:39 2019

@author: manpreetsaluja
"""

#import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


#import dataset
dataset=pd.read_csv("/Users/manpreetsaluja/Downloads/diabetes.csv")

dataset.describe()
dataset.head()

#impute 0 with nan
dataset.replace({'NaN':0})

#checking null values
print(dataset.isnull().sum())

