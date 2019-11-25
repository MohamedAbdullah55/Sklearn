from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error

'''
mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='
uniform_average')

mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='
uniform_average')

median_absolute_error(y_true, y_pred)

'''
y_tarin = [5,8,2,4,7,9]
y_pred =  [2,3,4,5,6,7]

print("By using mean_absolute_error with uniform_average")
MAE = mean_absolute_error(y_tarin, y_pred, multioutput='uniform_average')
print(MAE)

print("-------------------------------------------------")

print("By using mean_squared_error with uniform_average")
MSE = mean_squared_error(y_tarin, y_pred, multioutput='uniform_average')
print(MSE)

print("-------------------------------------------------")

print("By using median_absolute_error")
MDAE = median_absolute_error(y_tarin, y_pred)
print(MDAE)

print("__________________________________________________________________________")

y_tarin = [[5,8],
           [2,4],
           [7,9]]

y_pred =  [[2,3],
           [4,5],
           [6,7]]

print("By using mean_absolute_error with raw_values")
MAE = mean_absolute_error(y_tarin, y_pred, multioutput='raw_values')
print(MAE)

print("-------------------------------------------------")

print("By using mean_squared_error  with raw_values")
MSE = mean_squared_error(y_tarin, y_pred, multioutput='raw_values')
print(MSE)