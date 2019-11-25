import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error

#read data
dataset = load_boston()
X = dataset.data
y = dataset.target

print('data features number = ',str(X.shape[1]),' feature')
print("----------------------------------------------------------------------")

#select beast features
#model = LinearRegression(copy_X=True)
#FeatureSelectionMethod = SelectFromModel(estimator=model)
#X = FeatureSelectionMethod.fit_transform(X,y)
#
#print('new data features number = ',str(X.shape[1]),' feature')
#print("----------------------------------------------------------------------")

#normaize data
normalizer = StandardScaler(copy=True, with_mean=True, with_std=True)
X = normalizer.fit_transform(X)

#split data to train , valid , test
X_train_valid , X_test , y_train_valid , y_test = train_test_split(X, y, test_size=0.30, random_state = 55 , shuffle=True)
X_train , X_valid , y_train , y_valid = train_test_split(X, y, test_size=0.10, random_state = 55 , shuffle=True)

#model bulding
linearRegModel = LinearRegression(copy_X=True)
linearRegModel.fit(X_train,y_train)


#Calculating Details
print('Linear Regression Train Score is : ' , linearRegModel.score(X_train, y_train))
print('Linear Regression Test Score is : ' , linearRegModel.score(X_valid, y_valid))
print('Linear Regression Coef is : ' , linearRegModel.coef_)
print('Linear Regression intercept is : ' , linearRegModel.intercept_)
print("----------------------------------------------------------------------")

#predict
y_pred = linearRegModel.predict(X_test)

print("true , predicted")
for true , predicted in zip(y_pred[:5],y_test[:5]):
    print(true , predicted)

print("----------------------------------------------------------------------")

MAE = mean_absolute_error(y_test,y_pred)
print("mean_absolute_error = ",MAE)

print("----------------------------------------------------------------------")

MSE = mean_squared_error(y_test,y_pred)
print("mean_squared_error = ",MSE) 
