import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error

#load data
dataSet = load_iris()
X       = dataSet.data
y       = dataSet.target

print('X dimentions = ',X.shape)
print('y dimentions = ',y.shape)

#data scaling
normalizer = StandardScaler(with_mean=True, with_std=True, copy=True)
X          = normalizer.fit_transform(X)

#data spliting
skf_model = StratifiedKFold(n_splits = 5, random_state=55 ,shuffle=True)

#model bulding and runing
predict_1 = np.zeros(y.shape[0])
predict_2 = np.zeros(y.shape[0])
predict_3 = np.zeros(y.shape[0])
predict_4 = np.zeros(y.shape[0])
predict_5 = np.zeros(y.shape[0])
predict_6 = np.zeros(y.shape[0])
predict_7 = np.zeros(y.shape[0])

folds = skf_model.split(X,y)

for train_index , test_index in folds:
    X_train = X[train_index]
    X_test  = X[test_index]
    
    y_train = y[train_index]
    y_test  = y[test_index]
    
    LogisticRegression_1 = LogisticRegression(penalty='l2', solver='newton-cg', random_state = 55) 
    LogisticRegression_2 = LogisticRegression(penalty='l2', solver='sag', random_state = 55) 
    LogisticRegression_3 = LogisticRegression(penalty='l2', solver='lbfgs', random_state = 55) 
    LogisticRegression_4 = LogisticRegression(penalty='l1', solver='liblinear', random_state = 55)
    LogisticRegression_5 = LogisticRegression(penalty='l2', solver='liblinear', random_state = 55)
    LogisticRegression_6 = LogisticRegression(penalty='l1', solver='saga', random_state = 55)
    LogisticRegression_7 = LogisticRegression(penalty='l2', solver='saga', random_state = 55)
    
    LogisticRegression_1.fit(X_train,y_train)
    LogisticRegression_2.fit(X_train,y_train)
    LogisticRegression_3.fit(X_train,y_train)
    LogisticRegression_4.fit(X_train,y_train)
    LogisticRegression_5.fit(X_train,y_train)
    LogisticRegression_6.fit(X_train,y_train)
    LogisticRegression_7.fit(X_train,y_train)
    
    y_test_pred_1 = LogisticRegression_1.predict(X_test)
    y_test_pred_2 = LogisticRegression_2.predict(X_test)
    y_test_pred_3 = LogisticRegression_3.predict(X_test)
    y_test_pred_4 = LogisticRegression_4.predict(X_test)
    y_test_pred_5 = LogisticRegression_5.predict(X_test)
    y_test_pred_6 = LogisticRegression_6.predict(X_test)
    y_test_pred_7 = LogisticRegression_7.predict(X_test)
    
    predict_1[test_index] = y_test_pred_1
    predict_2[test_index] = y_test_pred_2
    predict_3[test_index] = y_test_pred_3
    predict_4[test_index] = y_test_pred_4
    predict_5[test_index] = y_test_pred_5
    predict_6[test_index] = y_test_pred_6
    predict_7[test_index] = y_test_pred_7
    
    print('Accuracy of model 1 \n' , accuracy_score(y_test,y_test_pred_1))
    print('Accuracy of model 2 \n' , accuracy_score(y_test,y_test_pred_2))
    print('Accuracy of model 3 \n' , accuracy_score(y_test,y_test_pred_3))
    print('Accuracy of model 4 \n' , accuracy_score(y_test,y_test_pred_4))
    print('Accuracy of model 5 \n' , accuracy_score(y_test,y_test_pred_5))
    print('Accuracy of model 6 \n' , accuracy_score(y_test,y_test_pred_6))
    print('Accuracy of model 7 \n' , accuracy_score(y_test,y_test_pred_7))
    print("----------------------------------------------------------------------")
    
print('total accuracy of model 1 \n',accuracy_score(y , predict_1))   
print('total accuracy of model 2 \n',accuracy_score(y , predict_2))  
print('total accuracy of model 3 \n',accuracy_score(y , predict_3))  
print('total accuracy of model 4 \n',accuracy_score(y , predict_4))  
print('total accuracy of model 5 \n',accuracy_score(y , predict_5))  
print('total accuracy of model 6 \n',accuracy_score(y , predict_6))  
print('total accuracy of model 7 \n',accuracy_score(y , predict_7)) 
print("----------------------------------------------------------------------")  
    
    
    
    
    