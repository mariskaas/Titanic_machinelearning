# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:02:13 2017

@author: mariska
"""
#Importing libraries to use
import pandas as pd
import numpy as np

print("Libraries have been imported")

#%%
#Reading in the training and testing files
titanic_training_data = pd.read_csv("C:/Users/mariska/Desktop/Coding_learning/Machine_learning_titanic/train_titanic.csv", index_col='PassengerId')
titanic_test_data =pd.read_csv("C:/Users/mariska/Desktop/Coding_learning/Machine_learning_titanic/test_titanic.csv", index_col='PassengerId')

print("Data has been loaded as pandas dataframe")

#%%
#Getting some basics on the data
print("Datatypes are:", titanic_training_data.dtypes)
print("Amount of NaNs are:", len(titanic_training_data.isnull()))
print("Colum names are:", titanic_training_data.columns)
