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

data = datanet[2000:]
data2 = datanet[:2000].drop('Churn', axis = 1)
data2.to_csv('datanet2.csv', sep=',')

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
data2['Dependents']=labelEncoder.fit_transform(data2['Dependents'])
data2['tenure']=labelEncoder.fit_transform(data2['tenure'])
data2['MultipleLines']=labelEncoder.fit_transform(data2['MultipleLines'])
data2['InternetService']=labelEncoder.fit_transform(data2['InternetService'])
data2['OnlineSecurity']=labelEncoder.fit_transform(data2['OnlineSecurity'])
data2['OnlineBackup']=labelEncoder.fit_transform(data2['OnlineBackup'])
data2['DeviceProtection']=labelEncoder.fit_transform(data2['DeviceProtection'])
data2['TechSupport']=labelEncoder.fit_transform(data2['TechSupport'])
data2['StreamingTV']=labelEncoder.fit_transform(data2['StreamingTV'])
data2['StreamingMovies']=labelEncoder.fit_transform(data2['StreamingMovies'])
data2['Contract']=labelEncoder.fit_transform(data2['Contract'])
data2['PaperlessBilling']=labelEncoder.fit_transform(data2['PaperlessBilling'])
data2['PaymentMethod']=labelEncoder.fit_transform(data2['PaymentMethod'])
data2['groupe_tenure']=labelEncoder.fit_transform(data2['groupe_tenure'])

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

x2 = data2[['gender','SeniorCitizen','Partner',
        'Dependents','tenure','MultipleLines',
        'InternetService','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV',
        'StreamingMovies','Contract','PaperlessBilling',
        'PaymentMethod','groupe_tenure']]

gb = GradientBoostingClassifier()

gb.fit(x, y)

y_pred = gb.predict(x2)
y_prob = gb.predict_proba(x2)

churnIndex = []

for i in range(0,len(data2)):
        if y_pred[i] == 1 :
                churnIndex.append(i)    


testData = pd.read_csv('datanet2.csv')

churned = []
churnedProv = {}

Churn = 0

for j in range(0, len(data2)):
        if j in churnIndex:
                churnedProv['customerID'] = testData['customerID'][j:j+1]
                churnedProv['synopsis'] = testData[j:j+1]['synopsis']
                churnedProv['film'] = testData[j:j+1]['film']
                churnedProv['Price'] = testData[j:j+1]['Price']
                churnedProv['Probability'] = y_prob[j][1] * 100
                churnedProv['Risk'] = ''

                if churnedProv['Probability'] > 50 and churnedProv['Probability'] < 65:
                        churnedProv['Risk'] = 'Low Churn Risk'
                        lowChurn += 1
                elif churnedProv['Probability'] >= 65 and churnedProv['Probability'] < 85:
                        churnedProv['Risk'] = 'Medium Churn Risk'
                        mediumChurn += 1
                else:
                        churnedProv['Risk'] = 'High Churn Risk'
                        highChurn += 1

                churned.append(churnedProv)
                churnedProv = {}

for k in range(0, len(churned)):
        print(churned[k]['customerID'].item(),
        "{0:.2f} %".format(churned[k]['Probability']),
        churned[k]['Risk'])

print("Low Churn Risk : {}".format(lowChurn))


model_name = 'model.pkl'
joblib.dump(gb, model_name)

