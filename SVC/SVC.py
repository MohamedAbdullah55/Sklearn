import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

'''
SVC(C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’, coef0=0.0, shrinking=True, probability=False, 
    tol=0.001, cache_size=200, class_weight=None,verbose=False, max_iter=-1, decision_function_shape=’ovr’, 
    random_state=None)
'''

#load data

dataSet = load_iris()
X       = dataSet.data
y       = dataSet.target
#dataSet = pd.read_csv("F:\\kaggel competition\\Classify forest types based on information about the area\\train.csv")
#dataSet = dataSet.drop(["Id"] , axis=1)
#X = dataSet.drop(["Cover_Type"] , axis=1)
#y = dataSet["Cover_Type"]

print('X dimentions = ',X.shape)
print('y dimentions = ',y.shape)

#data scaling
normalizer = StandardScaler(with_mean=True, with_std=True, copy=True)
X          = normalizer.fit_transform(X)

#data spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state = 55, shuffle=True)

#model bulding
svc_model = SVC(kernel= 'poly',degree=5, max_iter=10000,C=.5, random_state = 55)

svc_model.fit(X_train,y_train)

#model score
print('SVC Train Score is : ' , svc_model.score(X_train, y_train))
print('SVC Test Score is : ' , svc_model.score(X_test, y_test))

#model testing
y_test_pred= svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_test_pred)

print(cm)
sns.heatmap(cm)
plt.show()

plt.scatter(X_test[:,0],y_test)
plt.scatter(X_test[:,0],y_test_pred,color='r')
plt.show()
