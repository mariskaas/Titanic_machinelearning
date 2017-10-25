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
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

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
#Make some plots to check for outliers
a=titanic_training_data['Sex']
b=titanic_training_data['Age']
c=titanic_training_data['Fare']
d=titanic_training_data['Survived']

plt.scatter(d,b)
plt.show()

#%%
#Try to find out what we can fill age with if anything
titanic_training_data.groupby(['Pclass'], as_index=False).mean()
titanic_training_data.groupby(['Survived'], as_index=False).mean()
titanic_training_data.groupby(['Sex'], as_index=False).mean()
titanic_training_data.groupby(['Parch'], as_index=False).mean()

#%%#Most reliable seems fill with class averages of age, did not matter for classifier but apparently kaggle wants me to fill it so fine.
titanic_training_data.loc[(titanic_training_data['Age'].isnull()) & (titanic_training_data.Pclass == 1), 'Age']= 38.23
titanic_training_data.loc[(titanic_training_data['Age'].isnull()) & (titanic_training_data.Pclass == 2), 'Age']= 29.87
titanic_training_data.loc[(titanic_training_data['Age'].isnull()) & (titanic_training_data.Pclass == 3), 'Age']= 25.14
print(titanic_training_data['Age'])

#%%
#There are two datapoints with an exceptionally high fare, may be typo so remove
titanic_training_data = titanic_training_data[titanic_training_data['Fare']<500]
print("Outlier removed")
#%%
#Drop NaN values to make it workable for logarithm (subject to change according to features)
titanic_training_data_drop = titanic_training_data[['Sex', 'Age', 'Survived', 'Fare', 'Pclass', 'Parch']].dropna()
print("Data before drop:", len(titanic_training_data['Sex']))
print("Data after drop:", len(titanic_training_data_drop['Sex']))


#%%
#Setting labels and features (Subject to change)

labels = titanic_training_data_drop['Survived']

features = titanic_training_data_drop[['Sex', 'Fare', 'Pclass', 'Parch', 'Age']]
x=features
y=labels

print("Labels and features have been defined")
#%% Train, test split normal (1split)
x_train, x_test, y_train, y_test = train_test_split(
x, y, test_size=0.2, random_state=42)

print("Data has been split with a test size 0.2")

#%%
#First test a logistic regression
#Making the classifier
clf_logreg = LogisticRegression(class_weight = 'balanced')

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
print("Accuracy score logreg:", metrics.accuracy_score(y_pred, y_true))
print("Precision score logreg:", metrics.precision_score(y_pred, y_true))
print("Recall score logreg:", metrics.recall_score(y_pred, y_true))
print("f1 score logreg:", metrics.f1_score(y_pred, y_true))

#%% #Make a cross value logstic regression
clf_logreg_fold= LogisticRegression(class_weight = 'balanced')

#Predictions
scores_logreg = cross_val_score(clf_logreg_fold, x, y, cv=8, scoring='f1_macro')
#timing the SVM
t0=time()
print ("training time:", round(time()-t0, 3), "s")
print("F1 score mean for logreg is:", scores_logreg.mean())

#%%
#Trying a decision tree classifier
#making the classifier
clf_tree = DecisionTreeClassifier(criterion='entropy', min_samples_split=3)

#Fitting data and timing 
t0=time()
clf_tree.fit(x_train, y_train)
print ("Training time:", round(time()-t0, 3), "s")

#Predicing labels and timing
t0=time()
pred_tree = clf_tree.predict(x_test)
print ("Predicting time:", round(time()-t0, 3), "s")

y_pred = pred_tree
y_true = y_test
print("Accuracy score Tree:", metrics.accuracy_score(y_pred, y_true))
print("Precision score tree:", metrics.precision_score(y_pred, y_true))
print("Recall score Tree:", metrics.recall_score(y_pred, y_true))
print("f1 score Tree:", metrics.f1_score(y_pred, y_true))

#%%
#Trying the tree including the fold
clf_tree_fold = DecisionTreeClassifier(criterion = 'entropy', min_samples_split=7, random_state =42)

#Predictions
scores_tree = cross_val_score(clf_tree_fold, x, y, cv=8, scoring='f1_macro')

#Timing and result
t0=time()
print ("training time:", round(time()-t0, 3), "s")
print("F1 score mean for tree is:", scores_tree.mean())

#%% #Trying Random forest Classifier
#making the classififier
clf_random_forest = RandomForestClassifier(n_estimators=20, min_samples_split=9)

#Fitting data and timing 
t0=time()
clf_random_forest.fit(x_train, y_train)
print ("Training time:", round(time()-t0, 3), "s")

#Predicing labels and timing
t0=time()
pred_random_forest = clf_random_forest.predict(x_test)
print ("Predicting time:", round(time()-t0, 3), "s")

y_pred = pred_random_forest
y_true = y_test
print("Accuracy score forest:", metrics.accuracy_score(y_pred, y_true))
print("Precision score forest:", metrics.precision_score(y_pred, y_true))
print("Recall score forest:", metrics.recall_score(y_pred, y_true))
print("f1 score forest:", metrics.f1_score(y_pred, y_true))

#%%
#Random forest including folding
clf_random_forest_fold = RandomForestClassifier(n_estimators=20, min_samples_split=9)
#Predictions f1 macro
scores_random_forest_fold_f1 = cross_val_score(clf_random_forest_fold, x, y, cv=9, scoring='f1_macro')
#Predictions accruacy
scores_random_forest_fold_accuracy = cross_val_score(clf_random_forest_fold, x, y, cv=9, scoring='accuracy')

#Timing and result f1
t0=time()
print ("training time:", round(time()-t0, 3), "s")
print("F1 score mean for forest is:", scores_random_forest_fold_f1.mean())

#Timing and result accuracy
t0=time()
print ("training time:", round(time()-t0, 3), "s")
print("Accuracy score mean for forest is:", scores_random_forest_fold_accuracy.mean())

#%%
#Prepare the test set for predictions
#Change male and female in 1 and 2
titanic_test_data = titanic_test_data.replace('male', 0)
titanic_test_data = titanic_test_data.replace('female', 1)
print(titanic_test_data['Sex'])
print("Males are 0, Females are 1")
#%%
#Make some plots to check for outliers
a=titanic_test_data['Sex']
b=titanic_test_data['Age']
c=titanic_test_data['Fare']
d=titanic_test_data['Pclass']
plt.scatter(a,b)
plt.show()
plt.scatter(b,c)
plt.show()
plt.scatter(a,c)
plt.show()
plt.scatter(a,d)
plt.show()
plt.scatter(a,b)
plt.show()
plt.scatter(b,d)
plt.show()
plt.scatter(c,d)
plt.show()
#No crazy outliers found
#%%
#Fill age NaN because Kaggle specifically wants me to not drop them
titanic_test_data.groupby(['Pclass'], as_index=False).mean()
#Fill with average age per Pclass
titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data.Pclass == 1), 'Age']= 41
titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data.Pclass == 2), 'Age']= 29
titanic_test_data.loc[(titanic_test_data['Age'].isnull()) & (titanic_test_data.Pclass == 3), 'Age']= 24
print(titanic_test_data['Age'])
#%%
#Set the features from the test set

features_test = titanic_test_data[['Sex', 'Age', 'Fare', 'Pclass', 'Parch']]
print(np.isinf(titanic_test_data['Parch'].values.any()))
print("Datatypes are:", titanic_test_data.dtypes)
print(titanic_test_data.info())


print("Test features have been set")
#%% #Random forest classifier to predict from test set
#making the classififier
clf_random_forest_final = RandomForestClassifier(n_estimators=20, min_samples_split=9)

#Fitting data and timing 
t0=time()
clf_random_forest_final.fit(x, y)
print ("Training time:", round(time()-t0, 3), "s")

#Predicing labels and timing
t0=time()
pred_random_forest_final = clf_random_forest_final.predict(features_test)
print ("Predicting time:", round(time()-t0, 3), "s")

y_pred = pred_random_forest_final

print(len(y_pred))

#%%
#Making a dataframe with the passengderIDs and the list of predictions
titanic_test_data_drop['Predictions'] = y_pred
titanic_test_data_drop_newindex = titanic_test_data_drop.reset_index()
titanic_predictions = titanic_test_data_drop_newindex[['PassengerId', 'Predictions']].copy()
print(titanic_predictions)