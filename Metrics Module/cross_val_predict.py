import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

'''
•	و هي تقوم بحساب نتيجة تطبيق الموديل علي البيانات , اي قيمة y_pred

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
CrossValidatePredictValues = cross_val_predict(decision_Tree_Model,X,y,cv=3)

# Showing Results
print('Real y values : \n', y[:20])
print('Cross Validate Prediction : \n', CrossValidatePredictValues[:20])

print("---------------------------------------------------")

