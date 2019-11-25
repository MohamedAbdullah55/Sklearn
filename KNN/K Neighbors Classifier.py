import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

'''
KNeighborsClassifier(n_neighbors=5, weights=’uniform’, algorithm=’auto’, leaf_size=30, p=2, metric=’
                     minkowski’, metric_params=None,n_jobs=None, **kwargs)
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
k_neighbors_classifier_model = KNeighborsClassifier(n_neighbors=11, weights='distance', algorithm='auto')
k_neighbors_classifier_model.fit(X_train,y_train)
y_train_pred = k_neighbors_classifier_model.predict(X_train)
y_test_pred  = k_neighbors_classifier_model.predict(X_test)

print("score of train \n",k_neighbors_classifier_model.score(X_train, y_train))
print("score of test \n",k_neighbors_classifier_model.score(X_test, y_test))

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

