import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

'''
RandomForestClassifier(n_estimators=’warn’, criterion=’gini’, max_depth=None,
                       min_samples_split=2, min_samples_leaf=1,min_weight_fraction_leaf=0.0,
                       max_features=’auto’,max_leaf_nodes=None,min_impurity_decrease=0.0,
                       min_impurity_split=None, bootstrap=True,oob_score=False,
                       n_jobs=None,random_state=None, verbose=0,warm_start=False, class_weight=None)
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

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=55, shuffle=True)

print("Train data dimentions \n",X_train.shape)
print("Test data dimentions \n",X_test.shape)

print("---------------------------------------------------")

#model bulding
randome_forest_model = RandomForestClassifier(n_estimators=100, max_depth=3, criterion='gini', random_state=55)
randome_forest_model.fit(X_train,y_train)
y_train_pred = randome_forest_model.predict(X_train)
y_test_pred  = randome_forest_model.predict(X_test)

print("score of train \n",randome_forest_model.score(X_train, y_train))
print("score of test \n",randome_forest_model.score(X_test, y_test))

print("---------------------------------------------------")

cm_train = confusion_matrix(y_train, y_train_pred)
print("confusion matrix of train \n",cm_train)
sns.heatmap(cm_train)
plt.show()

cm_test = confusion_matrix(y_test, y_test_pred)
print("confusion matrix of test \n",confusion_matrix(y_test, y_test_pred))
sns.heatmap(cm_test)
plt.show()

print("---------------------------------------------------")

plt.scatter(X_test[:,0:1], y_test, color='red', s=150, label='Test data')
plt.scatter(X_test[:,0:1], y_test_pred, color='green', label='Prediction')
plt.legend()
plt.show()

