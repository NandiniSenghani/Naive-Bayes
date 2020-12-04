# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 21:41:19 2020

@author: Nandini senghani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import seaborn as sns

Salary_train = pd.read_csv("SalaryData_Train.csv")
Salary_test = pd.read_csv("SalaryData_Test.csv")
Salary_train1=pd.DataFrame(Salary_test)
Salary_train1= Salary_train1.append(Salary_train)
Salary_train.isnull().sum()
Salary_train.isna().sum()
Salary_test.isnull().sum()
Salary_test.isna().sum()
Salary_train.describe()
Salary_test.describe()
Salary_train.columns
#checking for outliers
plt.boxplot(Salary_train1.age,1,"ro",1)
plt.boxplot(Salary_train1.educationno,1,"ro",1)
plt.boxplot(Salary_train1.capitalgain,1,"ro",1)
plt.boxplot(Salary_train1.capitalloss,1,"ro",1)
plt.boxplot(Salary_train1.hoursperweek,1,"ro",1)
plt.hist(Salary_train1.education)
plt.hist(Salary_train1.workclass)
plt.hist(Salary_train1.maritalstatus)
plt.hist(Salary_train1.occupation)
plt.hist(Salary_train1.relationship)
plt.hist(Salary_train1.race)
plt.hist(Salary_train1.sex)
plt.hist(Salary_train1.Salary)
Q1=Salary_train1.quantile(0.25)
Q3=Salary_train1.quantile(0.95)
IQR = Q3-Q1
p= (Salary_train1<(Q1-1.5 * IQR))|(Salary_train1>(Q3-1.5*IQR))
p
Salary_train_out=Salary_train1[~( (Salary_train1<(Q1-1.5 * IQR))|(Salary_train1>(Q3-1.5*IQR))).any(axis=1)]
Salary_train_out.shape
#Here before removing the outliers we need to check  the entiers 
#if we replace the values with median it might not give better accuracy and analysis will not be up to the mark.

string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    Salary_train[i] = number.fit_transform(Salary_train[i])
    Salary_test[i] = number.fit_transform(Salary_test[i])

colnames = Salary_train.columns
len(colnames[0:13])
trainX = Salary_train[colnames[0:13]]
trainY = Salary_train[colnames[13]]
testX  = Salary_test[colnames[0:13]]
testY  = Salary_test[colnames[13]]

sgnb = GaussianNB()
smnb = MultinomialNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_gnb)
print ("Accuracy",(10759+1209)/(10759+601+2491+1209)) # 80%

spred_mnb = smnb.fit(trainX,trainY).predict(testX)
confusion_matrix(testY,spred_mnb)
print("Accuracy",(10891+780)/(10891+780+2920+780))  # 75%
