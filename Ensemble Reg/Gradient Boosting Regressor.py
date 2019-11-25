import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

'''
GradientBoostingRegressor(loss=’ls’, learning_rate=0.1,n_estimators=100, subsample=1.0, 
                          criterion=’friedman_mse’,min_samples_split=2,min_samples_leaf=1,
                          min_weight_fraction_leaf=0.0,max_depth=3,min_impurity_decrease=0.0,
                          min_impurity_split=None,init=None, random_state=None,max_features=None, 
                          alpha=0.9,verbose=0, max_leaf_nodes=None,warm_start=False, presort=’auto’, 
                          validation_fraction=0.1,n_iter_no_change=None, tol=0.0001)

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
gradient_boosting_Model = GradientBoostingRegressor(loss='ls', max_depth=2, random_state=55, learning_rate=0.05)
gradient_boosting_Model.fit(X_train,y_train)
y_train_pred = gradient_boosting_Model.predict(X_train)
y_test_pred  = gradient_boosting_Model.predict(X_test)

print("mean absolute error of train \n",mean_absolute_error(y_train, y_train_pred))
print("mean absolute error of test \n",mean_absolute_error(y_test, y_test_pred))

print("---------------------------------------------------")

print("mean squar error of train \n",mean_squared_error(y_train, y_train_pred))
print("mean squar error of test \n",mean_squared_error(y_test, y_test_pred))

print("---------------------------------------------------")

print("score of train \n",gradient_boosting_Model.score(X_train, y_train))
print("score of test \n",gradient_boosting_Model.score(X_test, y_test))

print("---------------------------------------------------")

X_train_axis = X_train[:,0:1]
y_train_axis = y_train

plt.scatter(X_train_axis, y_train_axis, color='red', label='train data')
plt.plot(X_train_axis, gradient_boosting_Model.predict(X_train), color='blue', label='fitting in train model')
plt.legend()
plt.show()

X_test_axis = X_test[:,0:1]
y_test_axis = y_test
plt.scatter(X_test_axis, y_test_axis, color='red', label='test data')
plt.plot(X_test_axis, gradient_boosting_Model.predict(X_test), color='green', label='fitting in test model')
plt.legend()
plt.show()
