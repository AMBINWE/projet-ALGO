import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

data = pd.read_csv('datanet')

data = datanet[2000:].drop('Churn', axis = 1)

labelEncoder = preprocessing.LabelEncoder()

data['synopsis']=labelEncoder.fit_transform(data['synopsis'])
data['film']=labelEncoder.fit_transform(data['film'])
data['Price']=labelEncoder.fit_transform(data['Price'])
data['Dependents']=labelEncoder.fit_transform(data['Dependents'])
data['customerID']=labelEncoder.fit_transform(data['customersID'])
data['MultipleLines']=labelEncoder.fit_transform(data['MultipleLines'])
data['InternetService']=labelEncoder.fit_transform(data['InternetService'])
data['OnlineSecurity']=labelEncoder.fit_transform(data['OnlineSecurity'])
data['OnlineBackup']=labelEncoder.fit_transform(data['OnlineBackup'])
data['DeviceProtection']=labelEncoder.fit_transform(data['DeviceProtection'])
data['TechSupport']=labelEncoder.fit_transform(data['TechSupport'])
data['StreamingTV']=labelEncoder.fit_transform(data['StreamingTV'])
data['StreamingMovies']=labelEncoder.fit_transform(data['StreamingMovies'])
data['Contract']=labelEncoder.fit_transform(data['Contract'])
data['PaperlessBilling']=labelEncoder.fit_transform(data['PaperlessBilling'])
data['PaymentMethod']=labelEncoder.fit_transform(data['PaymentMethod'])
data['Churn']=labelEncoder.fit_transform(data['Churn'])
data['groupe_tenure']=labelEncoder.fit_transform(data['groupe_tenure'])

x = data[['gender','SeniorCitizen','Partner',
        'Dependents','tenure','MultipleLines',
        'InternetService','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV',
        'StreamingMovies','Contract','PaperlessBilling',
        'PaymentMethod','groupe_tenure']]
y = data['Churn']

data['synopsis']=labelEncoder.fit_transform(data['synopsis'])
data['film']=labelEncoder.fit_transform(data['film'])
data['Price']=labelEncoder.fit_transform(data['Price'])

gb = GradientBoostingClassifier()

gb.fit(x, y)

y_pred = gb.predict(x2)
y_prob = gb.predict_proba(x2)

churnIndex = []

for i in range(0,len(data)):
        if y_pred[i] == 1 :
                churnIndex.append(i)    


testData = pd.read_csv('datanet2.csv')

churned = []
churnedProv = {}

Churn = 0

churnedProv['synopsis'] = testData[j:j+1]['synopsis']
print("Churn : {}".format(Churn))


model_name = 'model.pkl'
joblib.dump(gb, model_name)

