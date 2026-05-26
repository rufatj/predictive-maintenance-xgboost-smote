# -*- coding: utf-8 -*-


#!pip install xgboost
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost
url="https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
df=pd.read_csv(url)

#datalar
print(df.head())

#Boxplot graph
plt.figure(figsize=(6,6))
sns.boxplot(x=df['Machine failure'],color='green')
plt.title("Machine Failure Count")
plt.show()

#plt.subplots_adjust(hspace=1)  / just for seperating our graphs more clearly, but mainly don't needed

#Countplot Graph
plt.figure(figsize=(6,6))
sns.countplot(x="Machine failure",data=df)
plt.title("Machine Failure Count")
plt.show()

#data cleaning
df_clean=df.drop(columns=['UDI','Product ID'])
df_clean['Type']= df_clean['Type'].map({'L':0,'M':1,'H':2})
print("Our Final Cleaned Data Frame/DataSet ")
print(df_clean.head())

import re
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y=df_clean["Machine failure"]
# x=df_clean.drop["Machine failure"]  #but in our data set csv we have Failure Result, what's why we also have to remove them for measuring machine learning accuracy
x= df_clean.drop(columns=['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

x.columns=[re.sub(r'[\[\]<]','',col) for col in x.columns]

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=1)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
print("XGBoost model started")
model.fit(x_train, y_train)
print("Model successfully finished")

result=model.predict(x_test)  #equipment's predictions is result based on x values

print("----- MODEL REAL INDUSTRY RESULTS -----\n")
print(f"accuracy {accuracy_score(y_test,result)*100:.2f}%\n")

print("2. Classification Report:")
print(classification_report(y_test,result))

plt.figure(figsize=(6,6))

sns.heatmap(confusion_matrix(y_test, result), annot=True, fmt='d', cmap='Blues')
plt.xlabel("model's predictions")
plt.ylabel("real result")
plt.title("confusion matrix")
plt.show()

from numpy.random import random_sample
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score , confusion_matrix , classification_report
from imblearn.over_sampling import SMOTE

y=df_clean["Machine failure"]
x=df_clean.drop(columns=["Machine failure","TWF","HDF","PWF","OSF","RNF"])

x.columns=[re.sub(r'[\[\]<>]','',col) for col in x.columns]
x_train,x_test,y_train,y_test=train_test_split(x,y, train_size=0.8, random_state=1)

print(f'before the balancing our accident count {sum(y_train==1)}')
smote=SMOTE(random_state=1)
x_train_balanced,y_train_balanced= smote.fit_resample(x_train,y_train)
print(f"after smote accident count (Train): {sum(y_train_balanced == 1)}\n")

model_efficient = XGBClassifier(eval_metric='logloss', random_state=1)
print("XGBoost Efficient Model started...")
model_efficient.fit(x_train_balanced, y_train_balanced)
print("Model successfully finished!\n")

result_eff = model_efficient.predict(x_test)

print("----- EFFICIENT MODEL RESULTS WITH SMOTE -----\n")
print(f"New Accuracy: {accuracy_score(y_test, result_eff) * 100:.2f}%\n")
print(classification_report(y_test, result_eff))


plt.figure(figsize=(10, 5))
importances = model_efficient.feature_importances_
sns.barplot(x=importances, y=x.columns, palette="viridis", hue=x.columns, legend=False)
plt.title("Feature Importance Sensor")
plt.xlabel("Importantce degree")
plt.ylabel("Sensor names")
plt.show()

plt.figure(figsize=(6,6))
sns.heatmap(confusion_matrix(y_test, result_eff), annot=True, fmt='d', cmap='Oranges')
plt.xlabel("Model's Predictions")
plt.ylabel("Real Result")
plt.title("New Confusion Matrix (SMOTE)")
plt.show()
