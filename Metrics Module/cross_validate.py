import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

'''
•	و هي تقوم بحساب العديد من الارقام الهامة لاي موديل بعد عمل الfitting  مثل  : (fit time , test r2 , train r2)

'''

#load data
dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

print("Number of original features \n",X.shape[1])

#normalize data
normalizer = StandardScaler(with_mean=True, with_std=True, copy=True)
X = normalizer.fit_transform(X)

#features selection 
random_Forest_For_Features_Selection = RandomForestClassifier(n_estimators=100)
feature_selection = SelectFromModel(estimator=random_Forest_For_Features_Selection)
X = feature_selection.fit_transform(X, y)

print("Number of features \'after features selection\' \n",X.shape[1])

print("---------------------------------------------------")

#model bulding
decision_Tree_Model = DecisionTreeClassifier(criterion='gini', max_depth=2, splitter='best', random_state=55)

#cross_valid 
CrossValidateValues1 = cross_validate(decision_Tree_Model,X,y,cv=3,return_train_score = True)
CrossValidateValues2 = cross_validate(decision_Tree_Model,X,y,cv=3,scoring=('r2','neg_mean_squared_error'))

# Showing Results
print('Train Score Value : ', CrossValidateValues1['train_score'])
print('Test Score Value : ', CrossValidateValues1['test_score'])
print('Fit Time : ', CrossValidateValues1['fit_time'])
print('Score Time : ', CrossValidateValues1['score_time'])
print('Train MSE Value : ', CrossValidateValues2['train_neg_mean_squared_error'])
print('Test MSE Value : ', CrossValidateValues2['test_neg_mean_squared_error'])
print('Train R2 Value : ', CrossValidateValues2['train_r2'])
print('Test R2 Value : ', CrossValidateValues2['test_r2'])

print("---------------------------------------------------")

