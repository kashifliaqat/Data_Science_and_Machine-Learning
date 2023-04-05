# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 20:56:31 2023

@author: kashi
"""
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


#import data from local directory
df = pd.read_csv("healthcare-dataset-stroke-data.csv")



print(df.isnull().sum().sum())
df.dropna(inplace = True)
print(df.isnull().sum().sum())


