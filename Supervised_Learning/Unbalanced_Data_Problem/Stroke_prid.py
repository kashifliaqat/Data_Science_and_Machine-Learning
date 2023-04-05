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
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


#import data from local directory
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(df.isnull().sum().sum())
df.dropna(inplace = True)
print(df.isnull().sum().sum())


# Create a LabelEncoder object
le = LabelEncoder()
# Apply label encoding to the "gender" column
df['gender'] = le.fit_transform(df['gender'])
df['ever_married'] = le.fit_transform(df['ever_married'])
df['work_type'] = le.fit_transform(df['work_type'])
df['smoking_status'] = le.fit_transform(df['smoking_status'])
df['Residence_type'] = le.fit_transform(df['Residence_type'])

print(df)
print(df.isnull().sum().sum())

# Split the data into train and test sets
X = df.drop('stroke', axis=1)
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the classifiers and their parameter grids for GridSearchCV
rf = RandomForestClassifier(random_state=42)
rf_params = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10],'criterion': ['gini', 'entropy']}
dt = DecisionTreeClassifier(random_state=42)
dt_params = {'max_depth': [None, 5, 10, 15, 20], 'min_samples_split': [2, 5, 10], 'criterion': ['gini', 'entropy']}
lr = LogisticRegression(random_state=42)
lr_params = {'C': [0.1, 1, 10]}
knn = KNeighborsClassifier()
knn_params = {'n_neighbors': [3, 5, 7, 9]}

# Define the data techniques
oversampler = SMOTE(random_state=42)
undersampler = RandomUnderSampler(random_state=42)

# List to store the results
store_results = []
store_accuracies = []
from sklearn.impute import SimpleImputer




# Loop through different data techniques
for i, (name, sampler) in enumerate({'Original Data': None, 'SMOTE Data': oversampler, 'Undersampled Data': undersampler}.items()):
    if sampler is not None:
        X_train_res, y_train_res = sampler.fit_resample(X_train, y_train)
    else:
        X_train_res, y_train_res = X_train, y_train
    
    # Loop through different classifiers
    for j, (clf_name, clf_params) in enumerate({'Random Forest': (rf, rf_params), 'Decision Tree': (dt, dt_params),
                                                'Logistic Regression': (lr, lr_params), 'KNN': (knn, knn_params)}.items()):
        clf = clf_params[0]
        params = clf_params[1]
        # Perform GridSearchCV with 5-fold cross validation
        grid_clf = GridSearchCV(clf, params, cv=5)
        grid_clf.fit(X_train_res, y_train_res)
        best_clf = grid_clf.best_estimator_
        
        # Make predictions on test data
        y_pred = best_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)
        
        # Store the results
        store_results.append({'Data Technique': name, 'Classifier': clf_name, 'Best Params': grid_clf.best_params_,
                        'Validation Accuracy': grid_clf.best_score_, 'Test Accuracy': accuracy, 'Classification Report': classification_rep})
        # Store the accuracies only
        store_accuracies.append({'Data Technique': name, 'Classifier': clf_name, 'Validation Accuracy': grid_clf.best_score_, 'Test Accuracy': accuracy})

        print(f"Finished {clf_name} with {name}")
        print('Test Accuracy', accuracy)
# Convert results to dataframe
results_df = pd.DataFrame(results)
accuracies_df = pd.DataFrame(accuracies)