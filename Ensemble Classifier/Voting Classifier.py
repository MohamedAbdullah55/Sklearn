import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score

'''
VotingClassifier(estimators, voting=’hard’, weights=None,n_jobs=None, flatten_transform=None)

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

#loading models for Voting Classifier
gradient_boosting_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.05, n_estimators=1000, max_depth=3)
randome_forest_model = RandomForestClassifier(n_estimators=100, max_depth=3, criterion='gini', random_state=55)
decision_Tree_Model = DecisionTreeClassifier(criterion='gini', max_depth=2, splitter='best', random_state=55)
svc_model = SVC(kernel= 'poly',degree=5, max_iter=10000,C=.5, random_state = 55)
LogisticRegression = LogisticRegression(penalty='l1', solver='liblinear', random_state = 55)
sgd_classifier = SGDClassifier(loss='modified_huber', penalty='l1',learning_rate='optimal',random_state=33, max_iter=1000)

models = [('gradient_boosting_model',gradient_boosting_model),('randome_forest_model',randome_forest_model),
          ('decision_Tree_Model',decision_Tree_Model),('svc_model',svc_model),
          ('LogisticRegression',LogisticRegression),('sgd_classifier',sgd_classifier)]
#model bulding
voting_classifier_model = VotingClassifier(voting='hard',estimators=models)
voting_classifier_model.fit(X_train,y_train)
y_train_pred = voting_classifier_model.predict(X_train)
y_test_pred  = voting_classifier_model.predict(X_test)

print("score of train \n",voting_classifier_model.score(X_train, y_train))
print("score of test \n",voting_classifier_model.score(X_test, y_test))

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

