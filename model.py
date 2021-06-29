import pandas as pd
churn=pd.read_excel(r'C:\Users\wii\Desktop\py\CHURNDATA.xlsx')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()
churn['CUS_Gender']= label_encoder.fit_transform(churn['CUS_Gender'])
churn['CUS_Marital_Status']= label_encoder.fit_transform(churn['CUS_Marital_Status'])
churn['Status']= label_encoder.fit_transform(churn['Status'])
churn['TAR_Desc']= label_encoder.fit_transform(churn['TAR_Desc'])
from sklearn.feature_selection import mutual_info_classif
churn=churn.drop(['CUS_Gender','CUS_Marital_Status','CUS_Customer_Since',],axis=1)
churn=churn.dropna()
X=churn.iloc[:,:-1]
y=churn.iloc[:,-1:]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y , test_size = 0.20,random_state=1234)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
mutual_info=mutual_info_classif(X_train,y_train)

mutual_info=pd.Series(mutual_info)
mutual_info.index=X_train.columns
mutual_info.sort_values(ascending=False)
from sklearn.feature_selection import SelectKBest
selten=SelectKBest(mutual_info_classif,k=10)
selten.fit(X_train.fillna(0),y_train)
X_train.columns[selten.get_support()]


X_train=X_train[['# total debit transactions for S2',
       '# total debit transactions for S3', 'total debit amount for S2',
       'total debit amount for S3', '# total credit transactions for S2',
       '# total credit transactions for S3', 'total credit amount for S3',
       'total debit amount', 'total debit transactions', 'total transactions']]

X_test=X_test[['# total debit transactions for S2',
       '# total debit transactions for S3', 'total debit amount for S2',
       'total debit amount for S3', '# total credit transactions for S2',
       '# total credit transactions for S3', 'total credit amount for S3',
       'total debit amount', 'total debit transactions', 'total transactions']]

#Training
#model = LogisticRegression()
#model = neighbors.KNeighborsClassifier()
model_churn = DecisionTreeClassifier(criterion='entropy', max_depth= 7)
#model = SVC(kernel='linear',  gamma = 10, C= 1)
#model= RandomForestClassifier()
model_churn.fit(X_train,y_train)


#Testing
predicted = model_churn.predict(X_test)
predicted


import pickle
pickle.dump(model_churn,open('model_churn.pkl','wb'))
modlel=pickle.load(open('model_churn.pkl','rb'))



