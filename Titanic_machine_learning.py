# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:02:13 2017

@author: mariska
"""
#Importing libraries to use
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from time import time

print("Libraries have been imported")

#%%
#Reading in the training and testing files
titanic_training_data = pd.read_csv("C:/Users/mariska/Desktop/Coding_learning/Machine_learning_titanic/train_titanic.csv", index_col='PassengerId')
titanic_test_data =pd.read_csv("C:/Users/mariska/Desktop/Coding_learning/Machine_learning_titanic/test_titanic.csv", index_col='PassengerId')

print("Data has been loaded as pandas dataframe")

#%%
#Getting some basics on the data
print("Datatypes are:", titanic_training_data.dtypes)
print("Colum names are:", titanic_training_data.columns)

#%%
#Change sex data to zeroes and ones to make it workable for the logarithm
titanic_training_data = titanic_training_data.replace('male', 0)
titanic_training_data = titanic_training_data.replace('female', 1)
print(titanic_training_data['Sex'])
print("Males are 0, Females are 1")

#%%
#Drop NaN values to make it workable for logarithm (subject to change according to features)
titanic_training_data_drop = titanic_training_data[['Sex', 'Age', 'Survived']].dropna()
print("Data before drop:", len(titanic_training_data['Age']))
print("Data after drop:", len(titanic_training_data_drop['Age']))


#%%
#Setting labels and features (Subject to change)

labels = titanic_training_data_drop['Survived']

features = titanic_training_data_drop[['Sex', 'Age']]
x=features
y=labels

print("Labels and features have been defined")
#%% Train, test split normal (1split)
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=42)

print("Data has been split with a test size 0.2")

#%%
#First test a linear regression
#Making the classifier
clf_logreg = LogisticRegression()

#Fitting data and timing 
t0=time()
clf_logreg.fit(x_train, y_train)
print ("Training time:", round(time()-t0, 3), "s")

#Predicing labels and timing
t0=time()
pred_logreg = clf_logreg.predict(x_test)
print ("Predicting time:", round(time()-t0, 3), "s")

y_pred = pred_logreg
y_true = y_test
print("Accuracy score:", metrics.accuracy_score(y_pred, y_true))
print("Precision score:", metrics.precision_score(y_pred, y_true))
print("Recall score:", metrics.recall_score(y_pred, y_true))
print("f1 score:", metrics.f1_score(y_pred, y_true))

