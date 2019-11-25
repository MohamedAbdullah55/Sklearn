import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error

'''
model_selection.RandomizedSearchCV(estimator, param_distributions,n_iter=10, scoring=None,fit_params=None,n_jobs=
                                   None,iid=’warn’, refit=True, cv=’warn’, verbose=0, pre_dispatch=‘2*n_jobs’,
                                   random_state=None,error_score=’raise-deprecating’, return_train_score=’warn’)

'''

print("Logistic Regression applaied on iris data set")

print("----------------------------------------------------------------------")

#load data
dataSet = load_breast_cancer()
X       = dataSet.data
y       = dataSet.target

print('X dimentions = ',X.shape)
print('y dimentions = ',y.shape)

#data scaling
normalizer = StandardScaler(with_mean=True, with_std=True, copy=True)
X          = normalizer.fit_transform(X)

#data spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 55, shuffle=True)

#model bulding and runing
LogisticRegression = LogisticRegression(random_state = 55) 
SelectedParameters = {'penalty':['l2'],
                      'C':[1,2,3,4,5,100,200,300,400],
                      'fit_intercept':[True,False],
                      'class_weight':[None,'balanced'],
                      'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
                      'max_iter':[100,500,1000]
                      
                      }
RandomSearchModel = RandomizedSearchCV(LogisticRegression,SelectedParameters, n_iter=10,cv = 5,return_train_score=True)
RandomSearchModel.fit(X, y)
sorted(RandomSearchModel.cv_results_.keys())
RandomSearchResult = pd.DataFrame(RandomSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', RandomSearchResult )
print('Best Score is :', RandomSearchModel.best_score_)
print('Best Parameters are :', RandomSearchModel.best_params_)
print('Best Estimator is :', RandomSearchModel.best_estimator_)

BestLogisticRegressionModel = RandomSearchModel.best_estimator_

#model score
print('Logistic Regression model  Train Score is : ' , BestLogisticRegressionModel.score(X_train, y_train))
print('Logistic Regression model  Test Score is : ' , BestLogisticRegressionModel.score(X_test, y_test))

#model testing
y_test_pred = BestLogisticRegressionModel.predict(X_test)

#model metrics on validation set
cm = confusion_matrix(y_test,y_test_pred)

print(cm)
print("model f1 score = ",f1_score(y_test,y_test_pred,average='micro'))
print("model MAE score = ",mean_absolute_error(y_test,y_test_pred))
print("model MSE score = ",mean_squared_error(y_test,y_test_pred))
sns.heatmap(cm, center = True)
plt.show()

print('Used classes : ',BestLogisticRegressionModel.classes_)






