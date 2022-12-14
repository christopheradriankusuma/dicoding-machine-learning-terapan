# -*- coding: utf-8 -*-
"""Brain Stroke Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mJHFmsJK_aTfG2QwpDBNchrc-6jKgMPj

# Data Loading

Data merupakan dataset [brain stroke prediction](https://www.kaggle.com/datasets/zzettrkalpakbal/full-filled-brain-stroke-dataset) dari kaggle
"""

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 400 ~/.kaggle/kaggle.json
!kaggle datasets download -d zzettrkalpakbal/full-filled-brain-stroke-dataset

!unzip full-filled-brain-stroke-dataset.zip

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('full_data.csv')
df.head()

"""# Analysis

## Checking null data and basic statistics
"""

df.info()

df.isnull().sum()

df.describe()

"""## Grouping ages"""

df.loc[df['age'] <= 22, 'age'] = 1
df.loc[(df['age'] > 22) & (df['age'] <= 30), 'age'] = 2
df.loc[(df['age'] > 30) & (df['age'] <= 60), 'age'] = 3
df.loc[df['age'] >= 61, 'age'] = 4
df['age'] = df['age'].astype('int')

categorical_columns = ['age', 'gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'stroke']
numerical_columns = ['avg_glucose_level', 'bmi']

"""## Data distribution"""

plt.subplots(3, 3, figsize=(20, 16))

for i, col in enumerate(categorical_columns):
  plt.subplot(3, 3, i + 1)
  if col == 'smoking_status':
    df.groupby(col).size().plot(kind='bar', rot=45)
  else:
    df.groupby(col).size().plot(kind='bar', rot=0)

"""# Data preparation

## Handling categorical data
"""

df['gender'] = [0 if i == 'Male' else 1 for i in df['gender']]
df['ever_married'] = [0 if i == 'No' else 1 for i in df['ever_married']]
df = pd.get_dummies(df, columns=['age', 'work_type', 'Residence_type', 'smoking_status'])
df.head()

"""## Splitting dataset"""

df.shape

from sklearn.model_selection import train_test_split

X = df.drop(columns=['stroke'])
y = df['stroke']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""## Normalizing numerical columns"""

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train[numerical_columns])

X_train[numerical_columns] = scaler.transform(X_train.loc[:, numerical_columns])
X_test[numerical_columns] = scaler.transform(X_test.loc[:, numerical_columns])

"""# Modelling"""

from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

models = {
    'AdaBoost': AdaBoostClassifier(random_state=42), 
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42), 
    'Decision Tree': DecisionTreeClassifier(random_state=42), 
    'SVM': SVC(random_state=42),
}

for i in models.keys():
  models[i].fit(X_train, y_train)

"""# Evaluation"""

from sklearn.metrics import classification_report, precision_score, recall_score, accuracy_score, f1_score, confusion_matrix

test_df = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

for i in models.keys():
  pred = models[i].predict(X_test)

  recall = recall_score(y_test, pred, zero_division=0)
  precision = precision_score(y_test, pred, zero_division=0)
  accuracy = accuracy_score(y_test, pred)
  f1 = f1_score(y_test, pred, zero_division=0)
  test_df.loc[i] = [accuracy, precision, recall, f1]

  print(i)
  print(classification_report(y_test, pred, zero_division=0))
  cm = confusion_matrix(y_test, pred)
  sns.heatmap(cm, annot=cm, fmt='d')
  plt.show()
  print('=' * 100)

test_df

"""# Modelling and Evaluation - 2"""

stroke_len = df['stroke'].value_counts()[1]
df2 = df[df['stroke'] == 0].sample(stroke_len, random_state=42).copy()
df2 = pd.concat([df2, df[df['stroke'] == 1]])
df2.shape

from sklearn.model_selection import train_test_split

X = df2.drop(columns=['stroke'])
y = df2['stroke']

X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=42)

X_train2[numerical_columns] = scaler.transform(X_train2.loc[:, numerical_columns])
X_test2[numerical_columns] = scaler.transform(X_test2.loc[:, numerical_columns])

models = {
    'AdaBoost': AdaBoostClassifier(random_state=42), 
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42), 
    'Decision Tree': DecisionTreeClassifier(random_state=42), 
    'SVM': SVC(random_state=42),
}

for i in models.keys():
  models[i].fit(X_train2, y_train2)

test_df = pd.DataFrame(index=models.keys(), columns=['Accuracy', 'Precision', 'Recall', 'F1-Score'])

"""## Test on current testing dataset"""

for i in models.keys():
  pred = models[i].predict(X_test2)

  recall = recall_score(y_test2, pred, zero_division=0)
  precision = precision_score(y_test2, pred, zero_division=0)
  accuracy = accuracy_score(y_test2, pred)
  f1 = f1_score(y_test2, pred, zero_division=0)
  test_df.loc[i] = [accuracy, precision, recall, f1]

  print(i)
  print(classification_report(y_test2, pred, zero_division=0))
  cm = confusion_matrix(y_test2, pred)
  sns.heatmap(cm, annot=cm, fmt='d')
  plt.show()
  print('=' * 100)

test_df

"""## Test on previous testing dataset"""

for i in models.keys():
  pred = models[i].predict(X_test)

  recall = recall_score(y_test, pred, zero_division=0)
  precision = precision_score(y_test, pred, zero_division=0)
  accuracy = accuracy_score(y_test, pred)
  f1 = f1_score(y_test, pred, zero_division=0)
  test_df.loc[i] = [accuracy, precision, recall, f1]

  print(i)
  print(classification_report(y_test, pred, zero_division=0))
  cm = confusion_matrix(y_test, pred)
  sns.heatmap(cm, annot=cm, fmt='d')
  plt.show()
  print('=' * 100)

test_df