import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

'''
DecisionTreeRegressor(criterion=’mse’, splitter=’best’, max_depth=None,
                      min_samples_split=2, min_samples_leaf=1,
                      min_weight_fraction_leaf=0.0, max_features=None,
                      random_state=None, max_leaf_nodes=None,
                      min_impurity_decrease=0.0,
                      min_impurity_split=None, presort=False)

'''

#load data
dataset = load_boston()
X = dataset.data
y = dataset.target

print("Number of original features \n",X.shape[1])

#normalize data
normalizer = StandardScaler(with_mean=True, with_std=True, copy=True)
X = normalizer.fit_transform(X) 

#features selection 
linear_Reg_For_Features_Selection = LinearRegression(copy_X=True)
feature_selection = SelectFromModel(estimator=linear_Reg_For_Features_Selection)
X = feature_selection.fit_transform(X, y)

print("Number of features \'after features selection\' \n",X.shape[1])

print("---------------------------------------------------")

#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=55, shuffle=True)

print("Train data dimentions \n",X_train.shape)
print("Test data dimentions \n",X_test.shape)

print("---------------------------------------------------")

#model bulding
decision_Tree_Model = DecisionTreeRegressor(criterion='mse', max_depth=2, splitter='best', random_state=55)
decision_Tree_Model.fit(X_train,y_train)
y_train_pred = decision_Tree_Model.predict(X_train)
y_test_pred  = decision_Tree_Model.predict(X_test)

print("mean absolute error of train \n",mean_absolute_error(y_train, y_train_pred))
print("mean absolute error of test \n",mean_absolute_error(y_test, y_test_pred))

print("---------------------------------------------------")

print("mean squar error of train \n",mean_squared_error(y_train, y_train_pred))
print("mean squar error of test \n",mean_squared_error(y_test, y_test_pred))

print("---------------------------------------------------")

print("score of train \n",decision_Tree_Model.score(X_train, y_train))
print("score of test \n",decision_Tree_Model.score(X_test, y_test))

print("---------------------------------------------------")

X_train_axis = X_train[:,0:1]
y_train_axis = y_train

plt.scatter(X_train_axis, y_train_axis, color='red', label='train data')
plt.plot(X_train_axis, decision_Tree_Model.predict(X_train), color='blue', label='fitting in train model')
plt.legend()
plt.show()

X_test_axis = X_test[:,0:1]
y_test_axis = y_test
plt.scatter(X_test_axis, y_test_axis, color='red', label='test data')
plt.plot(X_test_axis, decision_Tree_Model.predict(X_test), color='green', label='fitting in test model')
plt.legend()
plt.show()












