import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

'''
•	وهي عملية حفظ الموديل بعد ان تم عمل تدريب له , وكانه حفظ للـ weights  الخاصة به , فيمكن استخدامه و عمل توقع له لاحقا دون تضييع وقت مرة اخري في التدريب

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
GridSearchModel = GridSearchCV(LogisticRegression,SelectedParameters, cv = 5,return_train_score=True)
GridSearchModel.fit(X, y)
sorted(GridSearchModel.cv_results_.keys())
GridSearchResults = pd.DataFrame(GridSearchModel.cv_results_)[['mean_test_score', 'std_test_score', 'params' , 'rank_test_score' , 'mean_fit_time']]

# Showing Results
print('All Results are :\n', GridSearchResults )
print('Best Score is :', GridSearchModel.best_score_)
print('Best Parameters are :', GridSearchModel.best_params_)
print('Best Estimator is :', GridSearchModel.best_estimator_)

BestLogisticRegressionModel = GridSearchModel.best_estimator_

#save model
joblib.dump(BestLogisticRegressionModel,'BestLogisticRegressionModel_For_breastCancerData.sav')

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

########################################

savedmodel = joblib.load('BestLogisticRegressionModel_For_breastCancerData.sav')
print(y[0:20])
print("-----------------------------------------------")
savedmodel.predict(X[0:20])




